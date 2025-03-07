import os
import torch
import random
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from diffusers import StableDiffusionPipeline


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '30000'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()




def create_pipeline(config):
    from diffusers.pipelines.stable_diffusion import safety_checker
    def dummy_checker(self, clip_input, images):
        return images, [False for _ in images]

    safety_checker.StableDiffusionSafetyChecker.forward = dummy_checker
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        resume_download=True,
        local_files_only=True
    )
    pipe.load_lora_weights(config.lora_path_dict, weight_name="pytorch_lora_weights.safetensors")

    return pipe

def get_class_info(class_dict_path):
    class_ids = []
    class_names = []
    with open(class_dict_path, 'r') as f:
        for line in f.read().splitlines():
            class_id, class_name = line.split(' ', 1)
            class_ids.append(class_id)
            class_names.append(class_name)
    return class_ids, class_names

def distributed_generator(rank, world_size, config):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    pipe = create_pipeline(config).to(device)

    sampler = DistributedSampler(
        range(config.total_classes),
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    for idx in sampler:
        class_id = config.class_ids[idx]
        class_name = config.class_names[idx]
        generate_class_images(
            pipe=pipe,
            class_id=class_id,
            class_name=class_name,
            save_path=config.save_path,
            samples_per_class=config.samples_per_class,
            batch_size=config.batch_size,
            device=device
        )

    cleanup()


def generate_class_images(pipe, class_id, class_name, save_path, samples_per_class, batch_size, device):
    class_dir = os.path.join(save_path, class_id)
    os.makedirs(class_dir, exist_ok=True)

    pbar = tqdm(
        total=samples_per_class,
        desc=f'[Rank {dist.get_rank()}] {class_name}',
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
        disable=dist.get_rank() != 0
    )

    for batch_idx in range(0, samples_per_class, batch_size):
        captions = [class_name] * batch_size
        images = pipe(captions, guidance_scale=2.0).images

        for img_idx, image in enumerate(images):
            filename = f"{class_id}_{batch_idx + img_idx:05d}.png"
            image.save(os.path.join(class_dir, filename))

        pbar.update(batch_size)

    pbar.close()
class GenerationConfig:
    def __init__(self):
        self.model_name = "sd-legacy/stable-diffusion-v1-5"
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.lora_path_dict = os.path.join(current_dir, "training/output/lbGen/checkpoint-128/")
        self.class_dict_path = "classnames.txt"
        self.class_ids, self.class_names = get_class_info(self.class_dict_path)
        self.total_classes = len(self.class_ids)
        self.samples_per_class = 1300
        self.batch_size = 65
        self.save_path = "IN1K"
        self.seed = 42



def main():
    config = GenerationConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.environ.update({
        "NCCL_TIMEOUT": "9000000000",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    })

    world_size = torch.cuda.device_count()
    mp.spawn(
        distributed_generator,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()

