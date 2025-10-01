from src.data.loader import load_image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


class InstructBLIPWrapper:
    def __init__(self, model_id="Salesforce/instructblip-flan-t5-xl"):
        # Force CPU
        self.device = "cpu"

        # Load processor + model
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def _clean_output(self, output: str, prompt: str) -> str:
        """
        Remove the prompt if the model echoes it back.
        """
        output = output.strip()
        if output.lower().startswith(prompt.lower()):
            return output[len(prompt):].strip(" :.-\n")
        return output

    def run(self, image_filename, prompt="Identify, list and describe all OHS issues in this image in a concise manner",
            max_new_tokens=128, num_beams=4):
        image = load_image(image_filename)
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Beam search
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        )

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self._clean_output(decoded, prompt)



