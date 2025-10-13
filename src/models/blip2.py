from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

class BLIP2Wrapper:
    def __init__(self):
        print("Loading BLIP-2 model...")
        self.device = "cpu"
        self.model_name = "Salesforce/blip2-flan-t5-xl"
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float32).to(self.device)

    def run(self, image_path, prompt=None):
        """
        Generates reasoning-based captions or answers depending on the prompt.
        """
        image = Image.open(image_path).convert("RGB")
        question = prompt if prompt else "Describe any safety risks in this image."

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)

        response = self.processor.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

