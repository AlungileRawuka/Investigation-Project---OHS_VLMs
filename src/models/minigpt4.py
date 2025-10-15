import os
import torch
import random
import numpy as np
from PIL import Image
from io import BytesIO
import torch.backends.cudnn as cudnn
from typing import Union
from transformers import StoppingCriteriaList

# MiniGPT-4 imports
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION_Vicuna0,
    CONV_VISION_LLama2,
    StoppingCriteriaSub
)

# Required for registry
from minigpt4.datasets.builders import *  # noqa
from minigpt4.models import *  # noqa
from minigpt4.processors import *  # noqa
from minigpt4.runners import *  # noqa
from minigpt4.tasks import *  # noqa


class MiniGPT4Wrapper:
    """
    CPU-only wrapper class for MiniGPT-4 model inference.
    Compatible with the common OHS_VLMs Gradio interface.
    """

    def __init__(self, cfg_path: str = "eval_configs/minigpt4_eval.yaml"):
        print("Initializing MiniGPT-4 Wrapper (CPU-only)...")

        self.cfg_path = cfg_path
        self.device = "cuda"  # ✅ Force CPU use

        self._setup_seed(42 + get_rank())
        self._load_config()
        self._load_model()
        self._setup_chat()

        print(f"MiniGPT-4 initialized on {self.device} ✓")

    # ------------------------- Helper functions -------------------------

    def _setup_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    def _load_config(self):
        from argparse import Namespace
        args = Namespace(cfg_path=self.cfg_path, gpu_id=0, options=None)
        self.cfg = Config(args)

    def _load_model(self):
        model_cfg = self.cfg.model_cfg
        model_cls = registry.get_model_class(model_cfg.arch)
        self.model = model_cls.from_config(model_cfg)
        self.model.to("cuda")  # ✅ Ensure model is loaded to CPU
        print("MiniGPT-4 model loaded on GPU ✓")

    def _setup_chat(self):
        # Select conversation template
        conv_dict = {
            "pretrain_vicuna0": CONV_VISION_Vicuna0,
            "pretrain_llama2": CONV_VISION_LLama2
        }
        model_type = self.cfg.model_cfg.model_type
        self.CONV_VISION = conv_dict.get(model_type, CONV_VISION_Vicuna0)

        # Vision processor
        vis_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_cls = registry.get_processor_class(vis_cfg.name)
        self.vis_processor = vis_cls.from_config(vis_cfg)

        # Stop words
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(self.device) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # Chat interface
        self.chat = Chat(
            self.model,
            self.vis_processor,
            device=self.device,
            stopping_criteria=stopping_criteria
        )

    # ------------------------- Public Inference Method -------------------------

    def run(self, image_input: Union[str, BytesIO], prompt: str) -> str:
        """
        Run MiniGPT-4 on an image and prompt (CPU-only).

        Args:
            image_input: path to image or BytesIO object
            prompt: text prompt for analysis

        Returns:
            str: Model-generated textual response
        """
        try:
            # Load and preprocess image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")

            chat_state = self.CONV_VISION.copy()
            img_list = []
            self.chat.upload_img(image, chat_state, img_list)
            self.chat.encode_img(img_list)

            self.chat.ask(prompt, chat_state)
            response = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0,
                max_new_tokens=300,
                max_length=2000
            )[0]

            print(f"[MiniGPT-4 CPU]: {response}")
            return response

        except Exception as e:
            print(f"❌ MiniGPT-4 inference error: {e}")
            return f"Error: {e}"


# ------------------------- Local Test -------------------------
if __name__ == "__main__":
    model = MiniGPT4Wrapper(cfg_path="eval_configs/minigpt4_eval.yaml")
    result = model.run("cat.jpg", "Describe all occupational health and safety risks in this image.")
    print("Output:", result)
