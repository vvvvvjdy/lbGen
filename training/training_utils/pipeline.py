import torch
from TrainableSDPipeline import TrainableSDPipeline
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from peft import LoraConfig


def _load_diffusion_pipeline(model_path, model_name, revision, weight_dtype, args=None):
    if model_name == 'sd_1_5':
        pipeline = TrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_type=weight_dtype)

    else:
        raise NotImplementedError("This model is not supported yet")
    return pipeline


def load_pipeline(args, model_name, weight_dtype, is_D=False):
    # Load pipeline
    pipeline = _load_diffusion_pipeline(args.pretrain_model, model_name, args.revision, weight_dtype, args)

    scheduler_args = {}

    if args.scheduler == "DPM++":
         pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler == "DDPM":
        if "variance_type" in  pipeline.scheduler.config:
            variance_type =  pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    if args.full_finetuning:
         pipeline.unet.to(dtype=torch.float)
    else:
        pipeline.unet.to(dtype=weight_dtype)
    pipeline.vae.to(dtype=weight_dtype)
    pipeline.text_encoder.to(dtype=weight_dtype)
    # set grad
    # Freeze vae and text_encoder
    pipeline.vae.requires_grad_(args.tune_vae)
    pipeline.text_encoder.requires_grad_(args.tune_text_encoder)
    pipeline.unet.requires_grad_(False)

    # gradient checkpoint
    if args.gradient_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()
        if args.tune_text_encoder or args.train_text_encoder_lora:
            pipeline.text_encoder.gradient_checkpointing_enable()
            pipeline.text_encoder_2.gradient_checkpointing_enable() if hasattr(pipeline, "text_encoder_2") else None
    
    # set trainable lora
    pipeline = set_pipeline_trainable_module(args, pipeline, is_D=is_D)
    
    return pipeline

def set_pipeline_trainable_module(args, pipeline, is_D=False):
    if not args.full_finetuning:
        # Set correct lora layers
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        pipeline.unet.add_adapter(unet_lora_config)

    if args.train_text_encoder_lora and not is_D:
        text_encoder_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipeline.text_encoder.add_adapter(text_encoder_lora_config)

    return pipeline

def get_trainable_parameters(args, pipeline, is_D=False):
    # load unet parameters
    if args.full_finetuning:
        G_parameters = list(pipeline.unet.parameters())
    else:
        G_parameters = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    
    
    # load other parameters, not for D
    if not is_D:
        text_lora_parameters = []
        if args.tune_vae:
            G_parameters.extend(pipeline.vae.parameters())
        
        if args.tune_text_encoder:
            G_parameters.extend(pipeline.text_encoder.parameters())
        
        if args.train_text_encoder_lora:
            text_lora_parameters = list(filter(lambda p: p.requires_grad, pipeline.text_encoder.parameters()))
    
        return G_parameters, text_lora_parameters
    else:
        return G_parameters