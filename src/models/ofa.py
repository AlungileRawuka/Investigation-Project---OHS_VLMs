import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from typing import Union, Optional


class OFAWrapper:
    """
    Wrapper for the OFA image captioning model (CPU-only).
    Can be used directly or imported into a Gradio interface.
    """

    def __init__(self, model_dir: str = "./ofa-base", device: str = "cpu"):
        self.device = device
        self.model_dir = model_dir

        # Load tokenizer and model
        self.tokenizer = OFATokenizer.from_pretrained(model_dir)
        self.model = OFAModel.from_pretrained(model_dir, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _load_image(self, image_input: Union[str, BytesIO]) -> torch.Tensor:
        """Load and preprocess image into tensor."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def generate_caption(self, image_input: Union[str, BytesIO], prompt: Optional[str] = "what does the image describe?") -> str:
        """Generate a caption for the provided image."""
        image_tensor = self._load_image(image_input)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                patch_images=image_tensor,
                num_beams=5,
                max_length=16,
                no_repeat_ngram_size=3
            )

        caption = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption


# Standalone test mode
if __name__ == "__main__":
    model = OFAWrapper(model_dir="./ofa-base", device="cpu")
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    print("Generated caption:", model.generate_caption(image_url))
