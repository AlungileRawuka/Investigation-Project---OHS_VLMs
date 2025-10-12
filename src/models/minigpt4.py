import argparse
import os
import random
import torch
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from typing import Union
from io import BytesIO

from transformers import StoppingCriteriaList
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports needed for registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

print("setup args....")
def parse_args(mock_args=None):
    parser = argparse.ArgumentParser(description="MiniGPT4 CLI Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id to use.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the config file (optional).",
        default=None
    )
    if mock_args:
        args = parser.parse_args(mock_args)
    else:
        args = parser.parse_args()
    return args


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

print("=== MiniGPT4 CLI demo starting ===")
args = parse_args([
    "--cfg-path", "eval_configs/minigpt4_eval.yaml",
    "--gpu-id", "0"
])
cfg = Config(args)

setup_seed(cfg.run_cfg.seed + get_rank())

device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize model
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
print("Model loaded to GPU")

# Choose conversation template
conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
              'pretrain_llama2': CONV_VISION_LLama2}
CONV_VISION = conv_dict[model_config.model_type]

# Vision processor
vis_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)

# Stop criteria
stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# Chat
chat = Chat(model, vis_processor, device=device, stopping_criteria=stopping_criteria)
print("Chat ready!")

def main(image_input: Union[str, BytesIO], prompt: str):

    # Load image
    if isinstance(image_input, str):  # path
        image = Image.open(image_input).convert("RGB")
    else:  # BytesIO or file-like
        image = Image.open(image_input).convert("RGB")

    # Initialize conversation
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask(prompt, chat_state)
    response = chat.answer(conv=chat_state, img_list=img_list,
                            num_beams=1, temperature=1.0,
                            max_new_tokens=300, max_length=2000)[0]
    print(f"MiniGPT-4: {response}")


if __name__ == "__main__":
    main("cat.jpg", "Describe this image")
