import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ---------- Helper Functions ----------

def compute_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    try:
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    except ZeroDivisionError:
        return 0.0


def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure


def compute_cosine(reference, candidate):
    vect = TfidfVectorizer().fit([reference, candidate])
    tfidf = vect.transform([reference, candidate])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


# ---------- Main Evaluation Function ----------

def evaluate(results_path, ground_truth_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.read_csv(results_path)
    gt_df = pd.read_csv(ground_truth_path)

    # Expect matching 'image' column for alignment
    merged = pd.merge(results_df, gt_df, on="image", how="inner")

    metrics = []
    for _, row in merged.iterrows():
        ref = str(row["ground_truth"]).strip()
        idefics_out = str(row["idefics_output"]).strip()
        blip_out = str(row["instructblip_output"]).strip()

        # Compute metrics for each model
        for model_name, output in [("idefics", idefics_out), ("instructblip", blip_out)]:
            bleu = compute_bleu(ref, output)
            rouge = compute_rouge(ref, output)
            cosine = compute_cosine(ref, output)

            metrics.append({
                "image": row["image"],
                "model": model_name,
                "BLEU": bleu,
                "ROUGE-L": rouge,
                "CosineSim": cosine
            })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)
    print(f"Saved metrics to {os.path.join(out_dir, 'evaluation_metrics.csv')}")

    # ---------- Aggregate Metrics ----------
    summary = metrics_df.groupby("model")[["BLEU", "ROUGE-L", "CosineSim"]].mean().reset_index()
    print("\n=== Summary Metrics ===")
    print(summary)

    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ---------- Visualization ----------
    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary["model"]))
    width = 0.25

    plt.bar(x - width, summary["BLEU"], width, label="BLEU")
    plt.bar(x, summary["ROUGE-L"], width, label="ROUGE-L")
    plt.bar(x + width, summary["CosineSim"], width, label="CosineSim")

    plt.xticks(x, summary["model"])
    plt.ylabel("Score")
    plt.title("Average Evaluation Metrics per Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_summary.png"))
    print(f"Saved plot to {os.path.join(out_dir, 'evaluation_summary.png')}")

    # ---------- Per-image Accuracy Bar ----------
    plt.figure(figsize=(10, 6))
    for metric in ["BLEU", "ROUGE-L", "CosineSim"]:
        pivot = metrics_df.pivot(index="image", columns="model", values=metric)
        pivot.plot(kind="bar", figsize=(10, 6))
        plt.title(f"{metric} per Image Comparison")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_per_image.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--ground_truth_csv", type=str, required=True, help="Path to ground_truth.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for metrics and plots")
    args = parser.parse_args()

    evaluate(args.results_csv, args.ground_truth_csv, args.out_dir)

