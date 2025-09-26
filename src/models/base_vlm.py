import torch
from PIL import Image

class BaseVLMWrapper:
    def __init__(self, model_name):
        self.model_name = model_name

    def run(self, image_path, prompt):
        

    def format_output(self, image_path, prompt, result):
        return {
            "model": self.model_name,
            "input_image": image_path,
            "prompt": prompt,
            "output": result
        }

