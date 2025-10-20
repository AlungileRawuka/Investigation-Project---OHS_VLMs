import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
import os

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

# === Safe CSV loader (handles Excel/non-UTF8 issues) ===
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception:
            return pd.read_csv(path, encoding_errors="ignore")

# === Load ground truth CSV ===
df_gt = safe_read_csv(ground_truth_csv)

# === Normalize filenames function ===
def normalize_filename(fname):
    return os.path.splitext(str(fname).strip().lower())[0]

df_gt["image_norm"] = df_gt["image"].apply(normalize_filename)

# === Initialize embedding model ===
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Function to compute mean metrics per model ===
def compute_mean_metrics(df_gt, model_csv, model_name):
    df_model = safe_read_csv(model_csv)
    df_model["image_norm"] = df_model["image"].apply(normalize_filename)

    # Merge only matching images
    merged = pd.merge(df_gt, df_model, on="image_norm", suffixes=("_gt", "_pred"))

    print(f"Total ground truth images: {len(df_gt)}")
    print(f"Total {model_name} predictions: {len(df_model)}")
    print(f"Matched images: {len(merged)}")

    meteor_scores, embedding_scores = [], []

    for _, row in merged.iterrows():
        gt_desc = str(row["output_gt"])
        pred_desc = str(row["output_pred"])

        # METEOR
        meteor_scores.append(meteor_score([gt_desc.split()], pred_desc.split()))

        # Embedding similarity
        emb_gt = embedding_model.encode(gt_desc, convert_to_tensor=True)
        emb_pred = embedding_model.encode(pred_desc, convert_to_tensor=True)
        embedding_scores.append(util.cos_sim(emb_gt, emb_pred).item())

    mean_meteor = sum(meteor_scores)/len(meteor_scores) if meteor_scores else 0
    mean_embedding = sum(embedding_scores)/len(embedding_scores) if embedding_scores else 0

    print(f"{model_name} -> METEOR: {mean_meteor:.3f}, Embedding Similarity: {mean_embedding:.3f}")
    return mean_meteor, mean_embedding

# === Compute metrics for all models ===
metrics_data = {}
for model_name, model_csv in model_files.items():
    metrics_data[model_name] = compute_mean_metrics(df_gt, model_csv, model_name)

# === Prepare data for plotting ===
models = list(metrics_data.keys())
metrics = ["METEOR", "Embedding Similarity"]
data = [[metrics_data[m][i] for m in models] for i in range(len(metrics))]

# === Plot grouped bar chart ===
x = range(len(models))
bar_width = 0.15

plt.figure(figsize=(6,6))  # reduced width
colors = ["#66b3ff", "#99ff99"]

for i, metric_name in enumerate(metrics):
    plt.bar([xi + (i - 0.5)*bar_width for xi in x],
            data[i],
            width=bar_width,
            label=metric_name,
            color=colors[i])

# === Formatting ===
plt.xticks(x, models, rotation=90, fontweight="bold", fontsize=11)  # vertical labels
plt.ylabel("Average Score", fontweight="bold", fontsize=12)
plt.xlabel("Models", fontweight="bold", fontsize=12)
plt.title("Comparison of Models", fontweight="bold", fontsize=14)
plt.ylim(0, 1)
plt.legend(title="Metrics", title_fontsize=11, fontsize=10, frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("model_comparison_bar_vertical.png", dpi=300)
plt.show()