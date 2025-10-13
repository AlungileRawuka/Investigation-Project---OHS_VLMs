from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPBaseWrapper:
    def __init__(self):
        print("Loading BLIP-Base model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to("cpu")

    def run(self, image_path, prompt=None):
        """
        Generate captions or OHS risk descriptions for the given image.
        If 'prompt' is provided, uses a structured Q&A prompt to guide the output.
        """
        image = Image.open(image_path).convert("RGB")

        if prompt and len(prompt.strip()) > 0:
            # --- Guided / reasoning mode (VQA-style) ---
            prompt_text = f"Question: {prompt.strip()} Answer:"
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to("cpu")
        else:
            # --- Pure captioning mode ---
            inputs = self.processor(images=image, return_tensors="pt").to("cpu")

        # Generate output with beam search for consistency
        out = self.model.generate(**inputs, max_length=80, num_beams=5, early_stopping=True)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

