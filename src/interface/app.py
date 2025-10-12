import gradio as gr
from PIL import Image
import pandas as pd
import os
import gc
import torch

# ---------------- Dynamic model loading ----------------
def load_model(model_name):
    if model_name == "IDEFICS":
        from src.models.idefics import IDEFICSWrapper
        return IDEFICSWrapper()
    elif model_name == "InstructBLIP":
        from src.models.instructblip import InstructBLIPWrapper
        return InstructBLIPWrapper()
    elif model_name == "BLIP-Base":
        from src.models.blip_base_wrapper import BLIPBaseWrapper
        return BLIPBaseWrapper()
    elif model_name == "BLIP-Large":
        from src.models.blip_large_wrapper import BLIPLargeWrapper
        return BLIPLargeWrapper()
    elif model_name == "BLIP-2":
        from src.models.blip2 import BLIP2Wrapper
        return BLIP2Wrapper()
    elif model_name == "LLaVA":
        from src.models.llava import LlavaWrapper
        return LlavaWrapper()
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ---------------- Prediction function ----------------
def predict_table(image, model_choice, prompt):
    if len(model_choice) == 0:
        return "<b style='color:red;'>Please select at least one model.</b>"
    if len(model_choice) > 2:
        return "<b style='color:red;'>Please select at most TWO models at a time to avoid memory overload.</b>"

    tmp_path = f"/tmp/{os.path.basename(getattr(image, 'filename', 'uploaded_image.png'))}"
    image.save(tmp_path)

    data = {"Model": [], "Output": []}

    for model_name in model_choice:
        try:
            print(f"\nLoading {model_name}...")
            model = load_model(model_name)
            output = model.run(tmp_path, prompt=prompt)
            data["Model"].append(model_name)
            data["Output"].append(output)
        except Exception as e:
            data["Model"].append(model_name)
            data["Output"].append(f"Error: {str(e)}")
        finally:
            # Free memory immediately after each model runs
            del model
            torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(data)
    return df.to_html(index=False, escape=False)

# ---------------- Gradio interface ----------------
model_options = ["IDEFICS", "InstructBLIP", "BLIP-Base", "BLIP-Large", "BLIP-2", "LLaVA"]

with gr.Blocks() as demo:
    gr.Markdown("## OHS Image Analysis – Compare Model Outputs")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        model_select = gr.CheckboxGroup(
            choices=model_options,
            label="Select up to 2 models to compare"
        )

    prompt_input = gr.Textbox(
        label="Enter analysis prompt",
        value="Identify, list and describe all OHS issues in this image in a concise manner"
    )

    output_html = gr.HTML(label="Model Outputs (side by side)")

    submit_btn = gr.Button("Run")
    submit_btn.click(
        predict_table,
        inputs=[image_input, model_select, prompt_input],
        outputs=output_html
    )

demo.launch(share=True)

