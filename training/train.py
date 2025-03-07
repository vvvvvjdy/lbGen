import json
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from transformers import AutoModelForCausalLM
import numpy as np
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from packaging import version
from peft import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel
from TrainableSDPipeline import TrainableSDPipeline


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from diffusers.optimization import get_scheduler


from training_utils.arguments import parse_args
from training_utils.logging import set_logger
from training_utils.pipeline import load_pipeline, get_trainable_parameters
from training_utils.gan_model import ProjectedDiscriminator
from training_utils.image_processing_clip_with_grad import CLIPImageProcessorWithGrad
from training_utils.image_processing_qalign_with_grad import QalignImageProcessorWithGrad



logger = get_logger(__name__, log_level="INFO")


class Trainer(object):

    def __init__(self, pretrained_model_name_or_path, args):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            total_limit=args.checkpoints_total_limit,
            logging_dir=logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        set_logger(args, self.accelerator, logger)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        print("weight_dtype", self.weight_dtype)
        self.model_name = args.pretrain_model_name
        self.pipeline = load_pipeline(args, self.model_name, self.weight_dtype)




        if args.version_clip_model == "openai/clip-vit-large-patch14":
            text_dim = 768

        elif args.version_clip_model == "openai/clip-vit-base-patch32":
            text_dim = 512

        else:
            raise ValueError("Unsupported CLIP model")
        self.D = ProjectedDiscriminator(
            c_dim=text_dim,
        )

        self.processor = AutoProcessor.from_pretrained(args.version_clip_model,local_files_only=True)
        self.image_processor_with_grad = CLIPImageProcessorWithGrad()

        self.clip_model = CLIPModel.from_pretrained(args.version_clip_model,local_files_only=True).to(dtype=self.weight_dtype, device=self.accelerator.device)
        self.clip_model.requires_grad_(False)

        self.q_align  = AutoModelForCausalLM.from_pretrained("q-future/one-align", trust_remote_code=True, attn_implementation="eager",
                                             torch_dtype=self.weight_dtype, device_map=self.accelerator.device)
        self.q_align.weight_tensor = torch.Tensor([5., 4., 3., 2., 1.]).to(dtype=self.weight_dtype, device=self.accelerator.device)
        self.q_image_processor_with_grad = QalignImageProcessorWithGrad()
        self.q_align.requires_grad_(False)




        self.global_step = 0
        self.first_epoch = 0
        self.resume_step = 0
        resume_global_step = 0
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                # assert False, "not implemented"
                path = os.path.basename(args.resume_from_checkpoint)
                load_dir = os.path.dirname(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                load_dir = args.output_dir

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                if args.full_finetuning:
                    self.pipeline.unet.load_state_dict(torch.load(os.path.join(load_dir, path, "unet.pt"), map_location="cpu"))
                else:
                    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(os.path.join(load_dir, path, "pytorch_lora_weights.safetensors"))
                    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=self.accelerator.unwrap_model(self.pipeline.unet))
                    if args.train_text_encoder_lora:
                        LoraLoaderMixin.load_lora_into_text_encoder(lora_state_dict, network_alphas=network_alphas, text_encoder=self.accelerator.unwrap_model(self.pipeline.text_encoder))

                if args.tune_vae:
                    self.pipeline.vae.load_state_dict(
                        torch.load(os.path.join(load_dir, path, "vae.pt"), map_location="cpu"))
                if args.tune_text_encoder:
                    self.pipeline.text_encoder.load_state_dict(
                        torch.load(os.path.join(load_dir, path, "text_encoder.pt"), map_location="cpu"))
                if args.resume_from_checkpoint == 'latest':
                    print("Loading D")
                    self.D.load_state_dict(torch.load(os.path.join(load_dir, path, "D.pt"), map_location="cpu"))

                self.global_step = int(path.split("-")[1])
                resume_global_step = self.global_step * args.gradient_accumulation_steps

        # load_trainable parameters, should be done after resume
        G_parameters, text_lora_parameters = get_trainable_parameters(args, self.pipeline, is_D=False)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Make sure the trainable params are in float32.
        if args.mixed_precision == "fp16":
            models = [self.pipeline.unet]
            if args.tune_vae:
                models.append(self.pipeline.vae)
            if args.tune_text_encoder or args.train_text_encoder_lora:
                models.extend([self.pipeline.text_encoder, self.pipeline.text_encoder_2])
            cast_training_params(models, dtype=torch.float32)

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        elif args.optimizer_class == 'AdamW':
            optimizer_cls = torch.optim.AdamW

        if args.train_text_encoder_lora:
            if args.textenc_lora_lr is None:  # share lr with text lora
                G_parameters.extend(text_lora_parameters)
                self.G_parameters = G_parameters
                self.optimizer = optimizer_cls(
                    self.G_parameters,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )
            else:
                self.optimizer = optimizer_cls([
                    dict(
                        params=G_parameters,
                        lr=args.learning_rate,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon),
                    dict(
                        params=text_lora_parameters,
                        lr=args.textenc_lora_lr,
                        betas=(args.adam_beta1, args.adam_beta2),
                        weight_decay=args.adam_weight_decay,
                        eps=args.adam_epsilon)
                ])
                self.G_parameters = G_parameters
                self.G_parameters.extend(text_lora_parameters)

        else:
            self.G_parameters = G_parameters
            self.optimizer = optimizer_cls(
                self.G_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        self.D_parameters = self.D.parameters()
        self.D_optimizer = optimizer_cls(
            self.D_parameters,
            lr=args.learning_rate_D,
            betas=(args.adam_beta1_D, args.adam_beta2_D),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # get train_dataset and train_dataloader
        self.train_dataset = load_dataset("dataset/imagenet.py", split="train", trust_remote_code=True)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers
        )
        self.D_train_dataset = load_dataset("dataset/imagenet.py", split="train", trust_remote_code=True)
        self.D_train_dataloader = torch.utils.data.DataLoader(
            self.D_train_dataset,
            shuffle=True,
            batch_size=int(args.train_batch_size*2),
            num_workers=args.dataloader_num_workers
        )

        # compute text embeddings
        class_names = self.D_train_dataset['class_name']
        texts = self.processor(text=class_names, padding=True, return_tensors="pt").to(self.accelerator.device)
        self.text_features = self.clip_model.get_text_features(**texts)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.first_epoch = self.global_step // self.num_update_steps_per_epoch
        self.resume_step = resume_global_step % (self.num_update_steps_per_epoch * args.gradient_accumulation_steps)

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        self.pipeline.to(self.accelerator.device)

        if args.tune_vae:
            self.pipeline.vae.to(dtype=torch.float)

        if args.tune_text_encoder:
            self.pipeline.text_encoder.to(dtype=torch.float)

        (self.pipeline.unet, self.optimizer, self.train_dataloader, self.lr_scheduler,
         self.D_optimizer, self.pipeline.text_encoder, self.D, self.D_train_dataloader ) = (
            self.accelerator.prepare(
            self.pipeline.unet, self.optimizer, self.train_dataloader, self.lr_scheduler,
                self.D_optimizer, self.pipeline.text_encoder, self.D,self.D_train_dataloader
        ))

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(args))
            none_keys = []
            for k, v in tracker_config.items():
                if v is None:
                    none_keys.append(k)

            for k in none_keys:
                tracker_config.pop(k)
            for k, v in tracker_config.items():
                print(f"{k}: {type(v)}")

            self.accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def train(self, args):

        # Train!
        total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {args.loops}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        global_step = self.global_step
        first_epoch = self.first_epoch
        resume_step = self.resume_step

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        def save_discriminator(output_dir):
            D = self.accelerator.unwrap_model(self.D)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(D.state_dict(), os.path.join(output_dir, "D.pt"))

        def save_and_evaluate_generator(output_dir, n_iter, save=True):
            unet = self.accelerator.unwrap_model(self.pipeline.unet)
            text_encoder_lora_layers = None
            if args.train_text_encoder_lora:
                text_encoder = self.accelerator.unwrap_model(self.pipeline.text_encoder)
                text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))

            if save:
                if args.full_finetuning:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save(unet.state_dict(), os.path.join(output_dir, "unet.pt"))
                else:
                    unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
                    LoraLoaderMixin.save_lora_weights(
                        save_directory=output_dir,
                        unet_lora_layers=unet_lora_layers,
                        text_encoder_lora_layers=text_encoder_lora_layers,
                    )
                if args.tune_vae:
                    torch.save(self.accelerator.unwrap_model(self.pipeline.vae).state_dict(), os.path.join(output_dir, "vae.pt"))

                if args.tune_text_encoder:
                    torch.save(self.accelerator.unwrap_model(self.pipeline.text_encoder).state_dict(), os.path.join(output_dir, "text_encoder.pt"))

            def dummy_checker(image, device, dtype):
                return image, None

            # Load previous pipeline
            self.pipeline.run_safety_checker = dummy_checker
            ori_scheduler = self.pipeline.scheduler
            ori_unet = self.pipeline.unet
            self.pipeline.unet = unet
            if args.train_text_encoder_lora:
                ori_text_encoder = self.pipeline.text_encoder
                self.pipeline.text_encoder = self.accelerator.unwrap_model(self.pipeline.text_encoder)

            if args.scheduler == "DPM++":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            elif args.scheduler == "DDPM":
                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in self.pipeline.scheduler.config:
                    variance_type = self.pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                self.pipeline.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config, **scheduler_args)

            if args.validation_prompts and args.num_validation_images > 0:
                if args.validation_prompts_file is not None:
                    with open(args.validation_prompts_file, 'r') as f:
                        val_prompts_from_file = f.readlines()
                    validation_prompts = args.validation_prompts + val_prompts_from_file
                    validation_prompts = [p.strip() for p in validation_prompts]
                else:
                    validation_prompts = args.validation_prompts
                generator = torch.Generator(device=self.accelerator.device).manual_seed(args.seed) if args.seed else None
                # avoid oom by shrinking bs
                all_images = [[] for _ in range(args.num_validation_images)]
                for start in range(0, len(validation_prompts), 1):
                    prompts = validation_prompts[start: start + 1]
                    with torch.autocast(device_type='cuda'):
                        images = [
                            self.pipeline(prompts, num_inference_steps=args.total_step, generator=generator,
                                          guidance_scale=args.cfg_scale, guidance_rescale=args.cfg_rescale).images
                            for _ in range(args.num_validation_images)
                        ]
                    for i, img in enumerate(images):
                        all_images[i].extend(img)

                images = all_images

                new_images = [[] for _ in validation_prompts]
                for image in images:
                    for i, img in enumerate(image):
                        new_images[i].append(img)

                for tracker in self.accelerator.trackers:
                    if tracker.name == "tensorboard":
                        for i, image in enumerate(new_images):
                            np_images = np.stack([np.asarray(img) for img in image])
                            tracker.writer.add_images(f"test_{i}", np_images, n_iter, dataformats="NHWC")

            self.pipeline.scheduler = ori_scheduler
            self.pipeline.unet = ori_unet
            if args.train_text_encoder_lora:
                self.pipeline.text_encoder = ori_text_encoder

        # evaluate before training
        if self.accelerator.is_main_process and not self.accelerator.is_last_process and global_step == 0 and resume_step == 0:
            with torch.no_grad():
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_and_evaluate_generator(save_path, global_step)
                torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        # evaluate after resume
        if self.accelerator.is_main_process and not self.accelerator.is_last_process and global_step % 100 == 0:
            with torch.no_grad():
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_and_evaluate_generator(save_path, global_step, save=False)
                torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        if args.do_classifier_free_guidance:
            # embed for pipeline
            with torch.no_grad():
                if isinstance(self.pipeline, TrainableSDPipeline):
                    null_embed = self.pipeline.encode_prompt("", self.accelerator.device, args.train_batch_size, do_classifier_free_guidance=False)[0]

                else:
                    raise NotImplementedError("This model is not supported yet")
        step_count = 0

        loss_D = torch.tensor(0.0)

        # we first train the discriminator before the loops for better guidance
        for epoch in range(args.d_epoch_start):
            logger.info(f"=================D_epoch {epoch + 1}=================")
            self.pipeline.unet.eval()
            self.D.train()
            self.pipeline.set_progress_bar_config(disable=True)
            train_loss = 0.0
            for step, batch in enumerate(self.D_train_dataloader):

                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # train discriminator
                with self.accelerator.accumulate(self.D):
                    with torch.no_grad():
                        images = self.pipeline.forward(prompt=batch["class_name"])
                        images = self.image_processor_with_grad(images=images, tensor_type="pt").to(
                            self.accelerator.device)
                        image_features = self.clip_model.get_image_features(**images)

                    batch_size = image_features.shape[0]
                    chosen_text_features = self.text_features[torch.randperm(batch_size), :]
                    loss = self.D(x=image_features, c=chosen_text_features, side='D')
                    loss_D = loss
                    step_count += 1

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    self.D_optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.D_parameters, args.max_grad_norm_D)
                    self.D_optimizer.step()

                logs = {"D_loss": loss.detach().item(),
                        "lr": args.learning_rate_D}

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    self.accelerator.log(logs, step=global_step)
                    train_loss = 0.0

                    logger.info(f"{global_step}: {json.dumps(logs, sort_keys=False, indent=4)}")

                progress_bar.set_postfix(**logs)

                self.accelerator.wait_for_everyone()

                if global_step >= args.max_train_steps:
                    break

        for loop_i in range(args.loops):

            logger.info(f"=================loop {loop_i+1}=================")
            # train diffusion model (we also simultaneously train the discriminator)
            step_count = 0

            self.pipeline.unet.train()
            if args.tune_text_encoder or args.train_text_encoder_lora:
                self.pipeline.text_encoder.train()
                self.pipeline.text_encoder_2.train() if hasattr(self.pipeline, "text_encoder_2") else None
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):

                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                total_step = args.total_step

                # train diffusion model
                with self.accelerator.accumulate(self.pipeline.unet):
                    # setting of backward
                    bp_on_trained = True
                    early_exit = False
                    double_laststep = False
                    fast_training = False

                    interval = total_step // args.K
                    max_start = total_step - interval * (args.K - 1) - 1
                    start = random.randint(0, max_start)
                    training_steps = list(range(start, total_step, interval))
                    detach_gradient = True

                    if args.tune_text_encoder or args.train_text_encoder_lora:
                        if isinstance(self.pipeline, TrainableSDPipeline):
                            null_embed = self.pipeline.encode_prompt("", self.accelerator.device, args.train_batch_size,
                                                                     do_classifier_free_guidance=False)[0]
                        else:
                            raise NotImplementedError("This model is not supported yet")



                    kwargs = dict(
                        prompt=batch["class_name"],
                        height=args.resolution,
                        width=args.resolution,
                        training_timesteps=training_steps,
                        detach_gradient=detach_gradient,
                        train_text_encoder=args.tune_text_encoder or args.train_text_encoder_lora,
                        num_inference_steps=total_step,
                        guidance_scale=args.cfg_scale,
                        guidance_rescale=args.cfg_rescale,
                        negative_prompt_embeds=null_embed if args.do_classifier_free_guidance else None,
                        early_exit=early_exit,
                        return_latents=False,
                    )

                    if isinstance(self.pipeline, TrainableSDPipeline):
                        images = self.pipeline.forward(bp_on_trained=bp_on_trained, double_laststep=double_laststep,
                                                       fast_training=fast_training, **kwargs)

                    else:
                        raise NotImplementedError("This model is not supported yet")

                    # we use clip to ensure the class of the generated image is the same as the prompt
                    prompts = []
                    for i in batch["class_name"]:
                        prompts.append(f"photo of {i}")
                    # clip reward for individual-image semantic alignment
                    clip_image_inputs = self.image_processor_with_grad(images=images, tensor_type="pt").to(
                        self.accelerator.device)
                    clip_text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(
                        self.accelerator.device)
                    image_features = self.clip_model.get_image_features(**clip_image_inputs)
                    text_features = self.clip_model.get_text_features(**clip_text_inputs)
                    # normalize features
                    image_features_n = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    text_features_n = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                    # compute clip loss
                    clip_loss = 1 - torch.matmul(image_features_n, text_features_n.t()).mean()
                    clip_loss = args.clip_reward_weight * clip_loss
                    step_count += 1

                    # q-align reward for quality assurance
                    q_image_inputs = self.q_image_processor_with_grad(images=images, tensor_type="pt")[
                        "pixel_values"].to(self.accelerator.device)
                    score = self.q_align.score(images=None, task_="quality", input_="image",
                                               image_tensor=q_image_inputs)
                    q_loss = args.q_align_reward_weight * (1 - score.mean() / 5)

                    # gan loss for entire-dataset semantic alignment
                    gan_loss = self.D(x=image_features, side="G")
                    gan_loss = args.gan_loss_weights * gan_loss

                    loss = gan_loss

                    norm = {}

                    def record_grad(grad):
                        norm['reward_norm'] = grad.norm(2).item()
                        if args.norm_grad:
                            grad = grad / (norm['reward_norm'] / 1e4)  # 1e4 for numerical stability
                        return grad

                    images.register_hook(record_grad)

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.G_parameters, args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # train discriminator simultaneously
                    with self.accelerator.accumulate(self.D):
                        batch_size = image_features.shape[0]
                        chosen_text_features = self.text_features[torch.randperm(batch_size), :]
                        loss_d = self.D(x=image_features.detach(), c=chosen_text_features.detach(), side='D')
                        loss_D = loss_d

                        self.D_optimizer.zero_grad()
                        self.accelerator.backward(loss_d)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.D_parameters, args.max_grad_norm_D)
                        self.D_optimizer.step()


                logs = {"lr": self.lr_scheduler.get_last_lr()[0],
                        "step_loss": loss.detach().item(),
                        }
                logs.update({'D_loss': self.accelerator.gather(loss_D.detach()).mean().item()})
                logs.update({'loss_en': self.accelerator.gather(gan_loss.detach()).mean().item()})
                logs.update({'loss_in': self.accelerator.gather(clip_loss.detach()).mean().item()})
                logs.update({'loss_q': self.accelerator.gather(q_loss.detach()).mean().item()})
                logs.update(norm)

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    self.accelerator.log(logs, step=global_step)
                    train_loss = 0.0

                    logger.info(f"{global_step}: {json.dumps(logs, sort_keys=False, indent=4)}")

                progress_bar.set_postfix(**logs)


                self.accelerator.wait_for_everyone()
                if global_step >= args.max_train_steps:
                    break

            if self.accelerator.is_main_process:
             with torch.no_grad():
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_discriminator(save_path)
                save_and_evaluate_generator(save_path, global_step)
             torch.cuda.empty_cache()
             if global_step >= args.max_train_steps:
                 break
        # Save the lora layers
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args.pretrain_model, args=args)
    trainer.train(args=args)