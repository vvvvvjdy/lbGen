import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrain_model", type=str, default="runwayml/stable-diffusion-v1-5", help="The pretrained model to use."
    )
    parser.add_argument(
        "--version_clip_model", type=str, default="openai/clip-vit-large-patch14", help="The clip model to use."
    )
    parser.add_argument(
        "--pretrain_model_name", type=str, default="sd_1_5", help="the name of pretrained model",
        choices=["sd_1_5", "sdxl", "sd_1_5_attrcon", "sdxl_unet", "sdxl_attrcon_unet", "sdxl_attrcon"]
    )
    parser.add_argument(
        "--clip_reward_weight",
        type=float,
        default=None,
        help="The weight of clip reward model.",
    )
    parser.add_argument(
        "--q_align_reward_weight",
        type=float,
        default=None,
        help="The weight of q-align reward model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help="A set of prompts evaluated every `--validation_steps` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--validation_prompts_file",
        type=str,
        default=None,
        help="A set of prompts evaluated every `--validation_steps` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint/refl",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--batch_repeat", type=int, default=1, help="Repeat the batch."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_D",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=False,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--enable_torch2_product", action="store_true", help="Whether or not to use torch2 product. \
            see https://huggingface.co/docs/diffusers/optimization/torch2.0"
    )
    
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="lbGen",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="The dimension of the LoRA update matrices.",
    )
    
    parser.add_argument(
        "--K",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="classifier free guidance scale for training and inference.",
    )

    parser.add_argument(
        "--cfg_rescale",
        type=float,
        default=0.0,
        help="cfg rescale.",
    )

    parser.add_argument(
        "--training_prompts",
        type=str,
        default="refl_data.txt",
    )

    parser.add_argument(
        "--image_folder",
        type=str
    )

    parser.add_argument(
        "--total_step",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default='DDPM',
        choices=["DDPM"]
    )

    parser.add_argument(
        "--bp_on_trained",
        action="store_true"
    )

    parser.add_argument(
        "--full_finetuning",
        action="store_true"
    )

    parser.add_argument(
        "--tune_vae",
        action="store_true"
    )
    
    parser.add_argument(
        "--tune_text_encoder",
        action="store_true"
    )
    parser.add_argument(
        "--train_text_encoder_lora",
        action="store_true"
    )
    parser.add_argument(
        "--textenc_lora_lr",
        type=float,
        default=None
    )
    parser.add_argument(
        "--norm_grad",
        action="store_true"
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gan_loss_weights",
        type=float,
        default=1.0,
    )
    parser.add_argument("--adam_beta1_D", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2_D", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm_D",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--gan_unet_lastlayer_cls",
        action='store_true'
    )

    parser.add_argument(
        "--optimizer_class",
        type=str,
        default='AdamW'
    )

    parser.add_argument(
        "--loops",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--d_epoch_start",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    args.do_classifier_free_guidance = args.cfg_scale > 1.0

    return args