from torchvision.transforms import transforms
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils import logging


logger = logging.get_logger(__name__)

# As we need to obtain gradients through CLIP, we modify the image processor to return gradients
class CLIPImageProcessorWithGrad(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.crop_size = 224
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str = "pt",
        **kwargs,
    ):
        images = self.transforms(images)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
