from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re

class LlavaWrapper:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        print(f"Loading LLaVA model (CPU only): {model_name} ...")
        self.device = "cpu"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

    def run(self, image_path, prompt=None):
        image = Image.open(image_path).convert("RGB")

        # Default, cleaner prompt
        base_prompt = prompt or "Describe the occupational hazards visible in this image."

        # Avoid adding redundant <image> tokens
        if "<image>" not in base_prompt:
            full_prompt = f"<image>\n{base_prompt.strip()}"
        else:
            full_prompt = base_prompt.strip()

        # Create input
        inputs = self.processor(
            text=[full_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        if "pixel_values" not in inputs or inputs["pixel_values"].nelement() == 0:
            raise ValueError("Image not processed correctly — zero image tokens found.")

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=180)
        raw_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # CLEANUP STEP ---
        # Remove the echoed prompt or any repeated question part
        cleaned = raw_text.strip()
        cleaned = re.sub(r"(?i).*?(describe|identify|list|explain).{0,80}:", "", cleaned)
        cleaned = cleaned.replace(base_prompt.strip(), "").strip()

        # Remove redundant spacing or newline patterns
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        # If output is empty after cleanup, fall back
        if not cleaned:
            cleaned = "(No valid output generated; model may have echoed the prompt.)"

        return cleaned



