import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from src.data.loader import load_image

class IDEFICSWrapper:
    def __init__(self, model_id="HuggingFaceM4/idefics2-8b"):
        # Force CPU
        self.device = "cpu"
        print("Using device:", self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)

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
        """
        Run inference on a single image (CPU).
        Returns generated text (post-processed to remove prompt echo).
        """
        # load and prepare images
        
        image = load_image(image_filename)


        # include <image> placeholder so text/images align
        text_prompt = f"<image>\n{prompt}"

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # Beam search generation
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,       # disable sampling for beam search
            early_stopping=True 
        )

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self._clean_output(decoded, prompt)





