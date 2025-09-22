from src.data.loader import load_image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


class InstructBLIPWrapper:
    def __init__(self, model_id="Salesforce/instructblip-vicuna-7b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def run(self, image_filename, prompt="Describe hazards in this scene."):
        """
        Run inference on a single image.
        Args:
            image_filename (str): Name of the image in data/raw/ (e.g., "lab1.jpg")
            prompt (str): Task instruction for the model
        Returns:
            str: Generated text description
        """
        image = load_image(image_filename, subdir="raw")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    model = InstructBLIPWrapper()
    result = model.run("sample_image.jpg")
    print("InstructBLIP output:", result)



