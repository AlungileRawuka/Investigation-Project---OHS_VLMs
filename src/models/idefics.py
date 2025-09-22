from src.data.loader import load_image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch


class IDEFICSWrapper:
    def __init__(self, model_id="HuggingFaceM4/idefics-9b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)

    def run(self, image_filename, prompt="Identify any OHS hazards."):
        """
        Run inference on a single image.
        Args:
            image_filename (str): Name of the image in data/raw/ (e.g., "workshop2.png")
            prompt (str): Instruction for IDEFICS
        Returns:
            str: Generated hazard description
        """
        image = load_image(image_filename, subdir="raw")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0]


if __name__ == "__main__":
    model = IDEFICSWrapper()
    result = model.run("sample_image.jpg")
    print("IDEFICS output:", result)

