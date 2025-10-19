import os
import time
import argparse
import pandas as pd
from src.models.llava import LlavaWrapper
from src.models.blip2 import BLIP2Wrapper  

def run_llava_blip2(img_root, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # 1. Load LLaVA model
    # -----------------------------
    print("\n🚀 Loading LLaVA model...")
    start_load = time.time()
    llava = LlavaWrapper()
    llava_load_time = time.time() - start_load
    print(f" LLaVA loaded in {llava_load_time:.2f} seconds")

    # -----------------------------
    # 2. Load BLIP2 model
    # -----------------------------
    print("\n🚀 Loading BLIP2 model...")
    start_load = time.time()
    blip2 = BLIP2Wrapper()
    blip2_load_time = time.time() - start_load
    print(f" BLIP2 loaded in {blip2_load_time:.2f} seconds")

    # -----------------------------
    # 3. Run Inference
    # -----------------------------
    llava_results = []
    blip2_results = []

    llava_start_time = time.time()
    blip2_start_time = time.time()

    for cls_name in sorted(os.listdir(img_root)):
        cls_path = os.path.join(img_root, cls_name)
        if not os.path.isdir(cls_path):
            continue

        print(f"\n Processing class: {cls_name}")

        images = sorted([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if limit:
            images = images[:limit]

        for img_file in images:
            img_path = os.path.join(cls_path, img_file)
            print(f"  🖼️ {img_file}")

            # --- Run LLaVA ---
            try:
                llava_out = llava.run(img_path)
            except Exception as e:
                llava_out = f"Error: {str(e)}"

            # --- Run BLIP2 ---
            try:
                blip2_out = blip2.run(img_path)
            except Exception as e:
                blip2_out = f"Error: {str(e)}"

            llava_results.append({
                "class": cls_name,
                "image": img_file,
                "output": llava_out
            })

            blip2_results.append({
                "class": cls_name,
                "image": img_file,
                "output": blip2_out
            })

    # Compute total inference times
    llava_infer_time = time.time() - llava_start_time
    blip2_infer_time = time.time() - blip2_start_time

    # -----------------------------
    # 4. Save Results
    # -----------------------------
    llava_csv = os.path.join(out_dir, "llava_results.csv")
    blip2_csv = os.path.join(out_dir, "blip2_results.csv")

    pd.DataFrame(llava_results).to_csv(llava_csv, index=False)
    pd.DataFrame(blip2_results).to_csv(blip2_csv, index=False)

    print(f"\n LLaVA results saved to {llava_csv}")
    print(f" BLIP2 results saved to {blip2_csv}")

    # -----------------------------
    # 5. Timing Summary
    # -----------------------------
    print("\nPerformance Summary")
    print(f"  LLaVA load time:        {llava_load_time:.2f} s")
    print(f"  LLaVA inference time:   {llava_infer_time:.2f} s")
    print(f"  BLIP2 load time:        {blip2_load_time:.2f} s")
    print(f"  BLIP2 inference time:   {blip2_infer_time:.2f} s")
    print(f"  Total runtime:          {llava_load_time + llava_infer_time + blip2_load_time + blip2_infer_time:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, default="data/images", help="Root directory with subfolders per class")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for CSVs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on images per class")
    args = parser.parse_args()

    run_llava_blip2(args.img_root, args.out_dir, args.limit)

