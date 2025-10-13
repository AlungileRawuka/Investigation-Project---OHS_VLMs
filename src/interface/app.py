# ================================
# 🚀 OHS VLM WebApp Interface (Clean + Futuristic Loader)
# ================================

import gradio as gr
from PIL import Image
import pandas as pd
import os
import gc
import torch
import time

# ----------------------------
# Dynamic Model Loader
# ----------------------------
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

# ----------------------------
# Prediction Function with Futuristic Loader
# ----------------------------
def predict_table(images, model_choice, prompt, progress=gr.Progress()):
    if len(model_choice) == 0:
        return "<b style='color:red;'>Please select at least one model.</b>", None
    if len(model_choice) > 2:
        return "<b style='color:red;'>Please select at most TWO models at a time to avoid memory overload.</b>", None

    if not isinstance(images, list):
        images = [images]

    data = {"Filename": [], "OHS_Issues": [], "Image_Path": []}
    total_steps = len(images) * len(model_choice)
    step = 0

    for img_file in images:
        img_path = f"/tmp/{os.path.basename(getattr(img_file, 'name', 'uploaded_image.png'))}"
        if isinstance(img_file, str):
            img_path = img_file
        else:
            with open(img_path, "wb") as f:
                f.write(img_file.read())

        for model_name in model_choice:
            step += 1
            progress(step/total_steps, desc=f"🤖 Processing {os.path.basename(img_path)} with {model_name}... ⚡")

            try:
                model = load_model(model_name)
                # Simulate loader activity
                time.sleep(1.5)
                output = model.run(img_path, prompt=prompt)
                data["Filename"].append(os.path.basename(img_path))
                data["OHS_Issues"].append(output)
                data["Image_Path"].append(img_path)
            except Exception as e:
                data["Filename"].append(os.path.basename(img_path))
                data["OHS_Issues"].append(f"Error: {str(e)}")
                data["Image_Path"].append(img_path)
            finally:
                del model
                torch.cuda.empty_cache()
                gc.collect()

    # ----------------------------
    # 🧾 Build HTML Table
    # ----------------------------
    html_rows = ""
    for i in range(len(data["Filename"])):
        img_html = f'<img src="{data["Image_Path"][i]}" width="200"/>'
        html_rows += f"<tr><td>{data['Filename'][i]}</td><td>{img_html}</td><td>{data['OHS_Issues'][i]}</td></tr>"

    html_table = f"""
    <div style="font-family: 'Segoe UI', sans-serif;">
        <table border="1" style="border-collapse:collapse;width:100%;text-align:left;">
            <tr style="background-color:#0a192f;color:#64ffda;">
                <th>Filename</th>
                <th>Image</th>
                <th>OHS Issues</th>
            </tr>
            {html_rows}
        </table>
    </div>
    """

    # ----------------------------
    # 📊 Generate CSV Report
    # ----------------------------
    df = pd.DataFrame({
        "Filename": data["Filename"],
        "OHS_Issues": data["OHS_Issues"],
        "Image_Path": data["Image_Path"]
    })
    csv_path = "/tmp/OHS_Report.csv"
    df.to_csv(csv_path, index=False)

    return html_table, csv_path

# ----------------------------
# 🌐 Gradio Interface
# ----------------------------
model_options = ["IDEFICS", "InstructBLIP", "BLIP-Base", "BLIP-Large", "BLIP-2", "LLaVA"]

with gr.Blocks(css="""
#title { 
    text-align: center; 
    font-size: 2em; 
    color: #64ffda; 
}
body { 
    background-color: #0a192f; 
}
.gradio-container { 
    font-family: 'Segoe UI', sans-serif; 
}
""") as demo:
    gr.Markdown("<h2 id='title'>🦾 OHS Image Analysis – AI Vision Models</h2>")

    with gr.Row():
        image_input = gr.Files(label="📂 Upload Image(s)", file_types=[".jpg", ".png", ".jpeg"])
        model_select = gr.CheckboxGroup(
            choices=model_options,
            label="🤖 Select up to 2 models to compare"
        )

    prompt_input = gr.Textbox(
        label="📝 Enter analysis prompt",
        value="Identify, list and describe all OHS issues in this image in a concise manner"
    )

    output_html = gr.HTML(label="🧠 Model Outputs (Side by Side)")
    download_csv = gr.File(label="⬇️ Download CSV Report")

    with gr.Row():
        submit_btn = gr.Button("🚀 Run Analysis", variant="primary")

    # Launch prediction
    submit_btn.click(
        predict_table,
        inputs=[image_input, model_select, prompt_input],
        outputs=[output_html, download_csv]
    )

demo.launch(share=True)
