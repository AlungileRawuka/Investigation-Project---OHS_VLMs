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
import base64
import traceback

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
    elif model_name == "OFA":
        from src.models.ofa import OFAWrapper
        return OFAWrapper()
    elif model_name == "MiniGPT-4":
        from src.models.minigpt4 import MiniGPT4Wrapper
        return MiniGPT4Wrapper()
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ----------------------------
# Convert image to base64 for HTML embedding
# ----------------------------
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None

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

    data = {"Filename": [], "OHS_Issues": [], "Image_Path": [], "Image_Base64": []}
    total_steps = len(images) * len(model_choice)
    step = 0

    for img_file in images:
        img_path = f"/tmp/{os.path.basename(getattr(img_file, 'name', 'uploaded_image.png'))}"
        if isinstance(img_file, str):
            img_path = img_file
        else:
            with open(img_path, "wb") as f:
                f.write(img_file.read())

        # Convert image to base64 for HTML display
        img_base64 = image_to_base64(img_path)

        for model_name in model_choice: 
            step += 1
            progress(step / total_steps, desc=f"🤖 Processing {os.path.basename(img_path)} with {model_name}... ⚡")

            model = None  # ensure variable exists
            try:
                print(f"\n=== Loading model: {model_name} ===")
                model = load_model(model_name)
                print(f"{model_name} successfully loaded on {model.device if hasattr(model, 'device') else 'unknown device'}")
                # Simulate loader activity

                time.sleep(1.5)
                print(f" Running model: {model_name}")
                output = model.run(img_path, prompt=prompt)
                print(f"{model_name} output: {output[:100]}...")  # show start of result for debug
                data["Filename"].append(os.path.basename(img_path))
                data["OHS_Issues"].append(output)
                data["Image_Path"].append(img_path)
                data["Image_Base64"].append(img_base64)
            except Exception as e:
                print(f"\n❌ ERROR in {model_name}:\n{e}")
                traceback.print_exc()  # full error trace in logs
                data["Filename"].append(os.path.basename(img_path))
                data["OHS_Issues"].append(f"Error: {str(e)}")
                data["Image_Path"].append(img_path)
                data["Image_Base64"].append(img_base64)
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                gc.collect()
    # ----------------------------
    # 🧾 Build HTML Table with clickable thumbnail images
    # ----------------------------
    html_rows = ""
    for i in range(len(data["Filename"])):
        if data["Image_Base64"][i]:
            img_html = f'''<img src="data:image/png;base64,{data["Image_Base64"][i]}" 
                          width="120" 
                          style="border-radius:8px; cursor:pointer; transition: transform 0.2s;" 
                          onclick="openModal(this.src)"
                          onmouseover="this.style.transform='scale(1.05)'"
                          onmouseout="this.style.transform='scale(1)'"/>'''
        else:
            img_html = '<span style="color:#ff6b6b;">Image not available</span>'
        html_rows += f"<tr><td>{data['Filename'][i]}</td><td>{img_html}</td><td>{data['OHS_Issues'][i]}</td></tr>"

    html_table = f"""
    <style>
        .modal {{
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            animation: fadeIn 0.3s;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(100, 255, 218, 0.3);
        }}
        
        .close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: #64ffda;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }}
        
        .close:hover {{
            color: #ff6b6b;
        }}
    </style>
    
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(imageSrc) {{
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = imageSrc;
            document.body.style.overflow = 'hidden';
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }}
        
        // Close modal with ESC key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
    
    <div style="font-family: 'Segoe UI', sans-serif;">
        <table border="1" style="border-collapse:collapse;width:100%;text-align:left;">
            <tr style="background-color:#0a192f;color:#64ffda;">
                <th>Filename</th>
                <th>Image Preview (Click to Enlarge)</th>
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
model_options = ["IDEFICS", "InstructBLIP", "BLIP-Base", "BLIP-Large", "BLIP-2", "LLaVA", "OFA", "MiniGPT-4"]

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
