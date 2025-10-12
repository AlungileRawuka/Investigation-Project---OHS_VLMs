import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from typing import Union

# Load tokenizer and model from local clone
MODEL_DIR = "./ofa-base"
tokenizer = OFATokenizer.from_pretrained(MODEL_DIR)
model = OFAModel.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def ofa_model(image_input: Union[str, BytesIO], prompt: str):
    
    if isinstance(image_input, str):  # path
        image = Image.open(image_input).convert("RGB")
    else:  # BytesIO or file-like
        image = Image.open(image_input).convert("RGB")
    # Preprocess image

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    # Run generation
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            patch_images=image_tensor,
            num_beams=5,
            max_length=16,
            no_repeat_ngram_size=3
        )

    # Decode result
    caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption
if __name__ == '__main__':
    print("Caption:", ofa_model("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"))