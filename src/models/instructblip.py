import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from src.data.loader import load_image
import re

class InstructBLIPWrapper:
    """
    Wrapper for Salesforce/instructblip-vicuna-7b
    Uses beam-based generation for better coherence and removes echo text.
    """

    def __init__(self, model_id="Salesforce/instructblip-vicuna-7b", device="cpu"):
        self.device = torch.device(device)
        print(f"Loading InstructBLIP model on device: {self.device}")

        # Load model and processor
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32  # CPU-friendly
        ).to(self.device)

        print("InstructBLIP successfully loaded on", self.device)

    def _clean_output(self, text: str, prompt: str) -> str:
        """
        Clean generated output — remove prompt echo, user/assistant tags, and noise.
        """
        text = re.sub(r"<image>|IMG_\d+\.\w+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?is)^.*?Assistant:\s*", "", text)
        text = re.sub(r"(?i)\bUser:\s*", "", text)

        # Remove echo of the original prompt
        if prompt.lower() in text.lower():
            text = re.sub(re.escape(prompt), "", text, flags=re.IGNORECASE)

        # Cleanup whitespace and extra punctuation
        text = re.sub(r"\s{2,}", " ", text)
        text = text.strip(" .:-\n")

        return text if text else "(No valid output generated; possibly CPU-limited)"

    def run(
        self,
        image_path: str,
        prompt: str = "Identify, list and describe all OHS issues in this image in a concise manner.",
        max_new_tokens: int = 256,
        num_beams: int = 3,
    ) -> str:
        """
        Run inference with beam search for more accurate and structured text.
        """
        image = load_image(image_path)

        # Prepare input tensors
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Beam-based generation for richer responses
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False,
            repetition_penalty=1.1,  # helps avoid repeated lines
            length_penalty=1.0        # balances brevity and detail
        )

        # Decode model output
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        return self._clean_output(decoded, prompt)




