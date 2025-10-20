import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
import os
import seaborn as sns
import chardet

# === Helper function to safely read CSV with encoding detection ===
def safe_read_csv(file_path):
    import chardet
    try:
        # First, try to detect encoding
        with open(file_path, "rb") as f:
            raw_data = f.read(50000)  # read a bigger chunk for better detection
            detected = chardet.detect(raw_data)
            encoding = detected["encoding"] or "utf-8"

        print(f"Trying to read '{file_path}' using encoding: {encoding}")
        return pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
    
    except UnicodeDecodeError:
        # Fallback to 'latin-1' if UTF-8/ascii fails
        print(f"UnicodeDecodeError detected. Reading '{file_path}' with 'latin-1'")
        return pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
    
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        raise

# === File paths ===
ground_truth_csv = "Ground_Truths.csv"
model_files = {
    "OFA": "OFA_results.csv",
    "MiniGPT-4": "MiniGPT4_results.csv",
    "blip2": "blip2_results.csv",
    "blip_base": "blip_base_results.csv",
    "blip_large": "blip_large_results.csv",
    "idefics": "idefics_results.csv",
    "instructblip": "instructblip_results.csv",
    "llava": "llava_results.csv"
}

# === Load ground truth CSV safely ===
df_gt = safe_read_csv(ground_truth_csv)
df_gt["image_norm"] = df_gt["image"].apply(lambda f: os.path.splitext(str(f).strip().lower())[0])

# === Initialize embedding model ===
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Compute metrics per model ===
all_scores = {}

for model_name, model_csv in model_files.items():
    df_model = safe_read_csv(model_csv)

    # Drop 'class' column from model CSV if it exists
    if "class" in df_model.columns:
        df_model = df_model.drop(columns=["class"])

    df_model["image_norm"] = df_model["image"].apply(lambda f: os.path.splitext(str(f).strip().lower())[0])

    # Merge with ground truth on normalized filenames
    merged = pd.merge(df_gt, df_model, on="image_norm", suffixes=("_gt", "_pred"))

    print(f"Model: {model_name}")
    print(f"Total ground truth images: {len(df_gt)}")
    print(f"Total model predictions: {len(df_model)}")
    print(f"Matched images: {len(merged)}\n")

    meteor_scores, embedding_scores, classes = [], [], []

    for _, row in merged.iterrows():
        gt_desc = str(row["output_gt"])
        pred_desc = str(row["output_pred"])

        # METEOR
        meteor_scores.append(meteor_score([gt_desc.split()], pred_desc.split()))

        # Embedding similarity
        emb_gt = embedding_model.encode(gt_desc, convert_to_tensor=True)
        emb_pred = embedding_model.encode(pred_desc, convert_to_tensor=True)
        embedding_scores.append(util.cos_sim(emb_gt, emb_pred).item())

        # Keep ground truth class
        classes.append(row["class"])

    merged["METEOR"] = meteor_scores
    merged["Embedding_Similarity"] = embedding_scores
    merged["class"] = classes
    all_scores[model_name] = merged

# === Prepare data for heatmap ===
class_list = sorted(df_gt["class"].unique())
metrics = ["METEOR", "Embedding_Similarity"]
heatmap_df = pd.DataFrame(index=class_list)

for model_name in model_files.keys():
    df = all_scores[model_name]
    for metric in metrics:
        mean_per_class = [df[df["class"] == cls][metric].mean() for cls in class_list]
        heatmap_df[f"{model_name}_{metric}"] = mean_per_class

# === Plot heatmap ===
plt.figure(figsize=(12, 6))

# Shorten x-axis labels by replacing full metric names with abbreviations
abbreviated_columns = [col.replace("METEOR", "Met").replace("Embedding_Similarity", "EmbSim") 
                       for col in heatmap_df.columns]
heatmap_df.columns = abbreviated_columns

# Create heatmap
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Score'})

# Bold title and axis labels
plt.title("Model Performance per Hazard Class(METEOR and Embending Similarity)", fontsize=14, fontweight="bold")
plt.xlabel("Model & Metric", fontsize=12, fontweight="bold")
plt.ylabel("Hazard Class", fontsize=12, fontweight="bold")

# Optional: rotate x-axis labels if crowded
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("model_class_heatmap.png", dpi=300)
plt.show()