# Investigation-Project---OHS_VLMs

This project investigates the application of **Vision-Language Models (VLMs)** for automatically identifying and describing **Occupational Health and Safety (OHS)** hazards on the University of the Witwatersrand campus. By leveraging multimodal reasoning between visual and textual inputs, the system aims to detect risks such as fall hazards, structural cracks, exposed wiring, and unclear emergency signage.

The research evaluates several VLM architectures, comparing their performance through quantitative metrics (e.g., cosine similarity and METEOR) and qualitative assessments to determine their reliability in real-world hazard detection scenarios.


**HARDWARE AND SOFTWARE REQUIREMENTS**

**Hardware**
**Minimum:**
CPU: 8 Core processor
RAM: 40GB 
Storage: 50GB of free space

**Recommended:**
CPU: 16 core or HPC node
The backend pipeline was run on the Wits cluster with 128 GB of CPU RAM

**SOFTWARE REQUIREMENTS**

-Operating System: Linux (tested on Wits HPC cluster dica10)

-Python: Version 3.8 or higher

-Dependencies: Listed in requirements.txt (includes PyTorch, Transformers, OpenCV, Flask/FastAPI, NumPy, and evaluation libraries)

-Git: For version control

**Reproducibilty Guide**"

-To reproduce results:

-Clone this repository and navigate to the project root.

-Set up the virtual environment and install dependencies (install requirements.txt).

-Run the interface or evaluation pipeline using:

python -m src.interface.app

This will produce the link to the web interface.

**To run the evaluation scripts**
Enter the following commands:

cd evaluation

python evaluation_bar_graph.py

python evaluation_heatmap.py

**MODEL SETUP: MiniGPT-4**
  - clone the repository: git clone https://github.com/Vision-CAIR/MiniGPT-4.git
  - change the directory to: cd MiniGPT-4
  - create python virtual enviroment: conda env create -f environment.yml
  - clone vicuna: git lfs install
                  git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
  - follow the instructions on: https://github.com/Vision-CAIR/MiniGPT-4 (for further setups like config files and model                                                                            weights).
  - include the script on: src/models/minigpt4.py (on the same folder)
  - run the script for inference: source /usr/local/etc/profile.d/conda.sh && conda activate minigptv && python minigpt4.py
  - for batch processing run: source /usr/local/etc/profile.d/conda.sh && conda activate minigptv && python batch_minigpt.py

**MODEL SETUP: OFA**
  - clone the repository: git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
  - install the model from huggingface:    git lfs install
                                           git clone https://huggingface.co/OFA-Sys/ofa-base
  - install transformers: cd OFA/transformers
                          !pip install .
  - include the script on: src/models/ofa.py  (on the same folder)
  - run the script for inference: python ofa.py
  - for batch processing run: python batch_ofa.py
