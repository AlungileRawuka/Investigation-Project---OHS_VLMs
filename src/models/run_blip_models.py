import os
import time
import argparse
import pandas as pd
from PIL import Image
from src.models.blip_base_wrapper import BLIPBaseWrapper
from src.models.blip_large_wrapper import BLIPLargeWrapper  

def run_blip_inference(img_root, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # 1. Load BLIP-Base
    # -----------------------------
    print("\n Loading BLIP-Base model...")
    start_load = time.time()
    blip_base = BLIPBaseWrapper()
    base_load_time = time.time() - start_load
    print(f"BLIP-Base loaded in {base_load_time:.2f} seconds")

    # -----------------------------
    # 2. Load BLIP-Large
    # -----------------------------
    print("\n Loading BLIP-Large model...")
    start_load = time.time()
    blip_large = BLIPLargeWrapper()
    large_load_time = time.time() - start_load
    print(f" BLIP-Large loaded in {large_load_time:.2f} seconds")

    # -----------------------------
    # 3. Run Inference
    # -----------------------------
    blip_base_results = []
    blip_large_results = []

    base_start_time = time.time()
    large_start_time = time.time()

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
            print(f"  🖼️  {img_file}")

            try:
                base_output = blip_base.run(img_path)
            except Exception as e:
                base_output = f"Error: {str(e)}"

            try:
                large_output = blip_large.run(img_path)
            except Exception as e:
                large_output = f"Error: {str(e)}"

            blip_base_results.append({
                "class": cls_name,
                "image": img_file,
                "output": base_output
            })

            blip_large_results.append({
                "class": cls_name,
                "image": img_file,
                "output": large_output
            })

    # Compute total inference times
    base_infer_time = time.time() - base_start_time
    large_infer_time = time.time() - large_start_time

    # -----------------------------
    # 4. Save Results
    # -----------------------------
    base_csv = os.path.join(out_dir, "blip_base_results.csv")
    large_csv = os.path.join(out_dir, "blip_large_results.csv")

    pd.DataFrame(blip_base_results).to_csv(base_csv, index=False)
    pd.DataFrame(blip_large_results).to_csv(large_csv, index=False)

    print(f"\n BLIP-Base results saved to {base_csv}")
    print(f" BLIP-Large results saved to {large_csv}")

    # -----------------------------
    # 5. Timing Summary
    # -----------------------------
    print("\n Performance Summary")
    print(f"  BLIP-Base load time:      {base_load_time:.2f} s")
    print(f"  BLIP-Base inference time: {base_infer_time:.2f} s")
    print(f"  BLIP-Large load time:     {large_load_time:.2f} s")
    print(f"  BLIP-Large inference time:{large_infer_time:.2f} s")
    print(f"  Total runtime:            {base_load_time + base_infer_time + large_load_time + large_infer_time:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, default="data/images", help="Root directory with subfolders per class")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for CSVs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on images per class")
    args = parser.parse_args()

    run_blip_inference(args.img_root, args.out_dir, args.limit)

