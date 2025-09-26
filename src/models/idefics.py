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

    def run(self, image_filename, prompt="Identify any OHS hazards.",
            max_new_tokens=128, temperature=0.0, num_beams=1, do_sample=False):
        """
        Run inference on a single image (CPU).
        Returns generated text (post-processed to remove prompt echo).
        """
        # load and prepare image
        image = load_image(image_filename, subdir="raw")

        # include <image> placeholder so text/images align
        text_prompt = f"<image>\n{prompt}"

        # Build inputs
        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt"
        )

        # Move inputs to device (CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Save input length to slice off prompt tokens later
        input_ids = inputs.get("input_ids", None)
        input_length = input_ids.shape[1] if input_ids is not None else 0

        # Generation kwargs
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=do_sample,
            early_stopping=True,
        )

        # Generate
        generated = self.model.generate(**inputs, **gen_kwargs)

        # generated is tensor shape (batch, seq_len). Remove prompt tokens:
        if input_length > 0:
            new_tokens = generated[:, input_length:]
        else:
            new_tokens = generated

        # Decode only the new tokens
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        return text


if __name__ == "__main__":
    model = IDEFICSWrapper(model_id="HuggingFaceM4/idefics2-8b")
    out = model.run("testImage.PNG",
                    prompt="Identify any OHS hazards and list them concisely (one per line). Do NOT repeat the prompt.",
                    max_new_tokens=120, temperature=0.0, num_beams=4, do_sample=False)
    print("Result:\n", out)


