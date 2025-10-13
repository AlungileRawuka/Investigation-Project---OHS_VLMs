from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPLargeWrapper:
    def __init__(self):
        print("Loading BLIP-Large model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to("cpu")

    def run(self, image_path, prompt=None):
        image = Image.open(image_path).convert("RGB")

        if prompt and len(prompt.strip()) > 0:
            # Use VQA mode with structured template
            prompt_text = f"Question: {prompt.strip()} Answer:"
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to("cpu")
        else:
            # Use captioning mode
            inputs = self.processor(images=image, return_tensors="pt").to("cpu")

        out = self.model.generate(**inputs, max_length=80, num_beams=5, early_stopping=True)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


