# app_gradio_table.py
import gradio as gr
from PIL import Image
from src.models.idefics import IDEFICSWrapper
from src.models.instructblip import InstructBLIPWrapper
import pandas as pd

# ---------------- Load models ----------------
idefics_model = IDEFICSWrapper()
instructblip_model = InstructBLIPWrapper()

# ---------------- Prediction function ----------------
def predict_table(image, model_choice, prompt):
    tmp_path = f"/tmp/{image.filename if hasattr(image, 'filename') else 'uploaded_image.png'}"
    image.save(tmp_path)

    data = {"Model": [], "Output": []}

    if "IDEFICS" in model_choice:
        data["Model"].append("IDEFICS")
        data["Output"].append(idefics_model.run(tmp_path, prompt=prompt))
    if "InstructBLIP" in model_choice:
        data["Model"].append("InstructBLIP")
        data["Output"].append(instructblip_model.run(tmp_path, prompt=prompt))

    df = pd.DataFrame(data)
    return df.to_html(index=False, escape=False)

# ---------------- Gradio interface ----------------
model_options = ["IDEFICS", "InstructBLIP"]

with gr.Blocks() as demo:
    gr.Markdown("##  OHS Image Analysis – Compare Model Outputs")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an image")
        model_select = gr.CheckboxGroup(
            choices=model_options, 
            value=model_options, 
            label="Select models"
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

