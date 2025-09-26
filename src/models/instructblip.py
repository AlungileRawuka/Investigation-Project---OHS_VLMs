from src.data.loader import load_image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


class InstructBLIPWrapper:
    def __init__(self, 
                 model_id="Salesforce/instructblip-flan-t5-xl",
                 max_new_tokens=128,
                 temperature=1.0,
                 num_beams=1,
                 do_sample=False):
        # Force CPU
        self.device = "cpu"

        # Load processor + model
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

        # Store generation kwargs
        self.gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=do_sample,
            early_stopping=True
        )

    def run(self, image_filename, prompt="Identify any OHS hazards and list them concisely (one per line)"):
        """
        Run inference on a single image.
        Args:
            image_filename (str): Name of the image in data/raw/ (e.g., "lab1.jpg")
            prompt (str): Task instruction for the model
        Returns:
            str: Generated text description
        """
        # Load image
        image = load_image(image_filename, subdir="raw")

        # Preprocess inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # Generate output
        output = self.model.generate(**inputs, **self.gen_kwargs)

        # Decode output
        return self.processor.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    model = InstructBLIPWrapper(max_new_tokens=100, temperature=0.7, num_beams=3, do_sample=True)
    result = model.run("testImage.PNG")
    print("InstructBLIP output:", result)




