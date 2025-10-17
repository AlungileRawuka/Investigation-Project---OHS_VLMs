import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from src.data.loader import load_image
import re

class IDEFICSWrapper:
    """
    Wrapper for HuggingFaceM4/Idefics3-8B-Llama3
    Clean, CPU-safe, and prompt-corrected.
    """

    def __init__(self, model_id="HuggingFaceM4/Idefics3-8B-Llama3", device="cpu"):
        self.device = torch.device(device)
        print(f"Loading IDEFICS model on device: {self.device}")

        # Load model + processor using Idefics3-specific class
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,   # safer for CPU
        ).to(self.device)

        print("IDEFICS successfully loaded on", self.device)

    def _clean_output(self, text: str, prompt: str) -> str:
        """
        Remove user/assistant labels, prompt echo, and extra noise.
        """

        # Remove "User:" and everything up to "Assistant:"
        text = re.sub(r"(?is)^.*?Assistant:\s*", "", text)

        # Remove any leftover "User:" label if it appears again
        text = re.sub(r"(?i)\bUser:\s*", "", text)

        # Remove the prompt itself if echoed
        if prompt.lower() in text.lower():
            pattern = re.escape(prompt)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up extra spaces and formatting
        text = re.sub(r"\s{2,}", " ", text)
        text = text.strip(" .:-\n")

        return text if text else "(No valid output generated; possibly CPU-limited)"

    def run(
        self,
        image_path: str,
        prompt: str = "Identify, list and describe all OHS issues in this image in a concise manner.",
        max_new_tokens: int = 128,
    ) -> str:
        """
        Run inference on an image and return model output.
        """
        image = load_image(image_path)

        # Structured chat format
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

        # Build inputs
        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = self.processor(
            images=[image],
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate output
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return self._clean_output(decoded, prompt)


