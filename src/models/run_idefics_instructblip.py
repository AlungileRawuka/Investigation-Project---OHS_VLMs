import os
import argparse
import pandas as pd
import time
from src.models.idefics import IDEFICSWrapper
from src.models.instructblip import InstructBLIPWrapper

def run_inference(img_root, out_dir, limit=10):
    os.makedirs(out_dir, exist_ok=True)

    # Get first N images from dataset
    all_images = sorted([
        f for f in os.listdir(img_root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    selected_images = all_images[:limit]

    # --- Load models and measure time ---
    start_time = time.time()
    idefics = IDEFICSWrapper()
    idefics_load_time = time.time() - start_time
    print(f"IDEFICS loaded in {idefics_load_time:.2f} seconds")

    start_time = time.time()
    instructblip = InstructBLIPWrapper()
    instructblip_load_time = time.time() - start_time
    print(f"InstructBLIP loaded in {instructblip_load_time:.2f} seconds")

    # Containers for results
    idefics_results = []
    instructblip_results = []

    for img_file in selected_images:
        img_path = os.path.join(img_root, img_file)
        print(f"\nProcessing {img_file}...")

        # IDEFICS inference timing
        start_time = time.time()
        try:
            idefics_out = idefics.run(img_path)
        except Exception as e:
            idefics_out = f"Error: {str(e)}"
        idefics_infer_time = time.time() - start_time
        print(f"IDEFICS inference time: {idefics_infer_time:.2f} seconds")

        idefics_results.append({
            "image": img_file,
            "output": idefics_out,
            "inference_time_sec": idefics_infer_time
        })

        # InstructBLIP inference timing
        start_time = time.time()
        try:
            blip_out = instructblip.run(img_path)
        except Exception as e:
            blip_out = f"Error: {str(e)}"
        instructblip_infer_time = time.time() - start_time
        print(f"InstructBLIP inference time: {instructblip_infer_time:.2f} seconds")

        instructblip_results.append({
            "image": img_file,
            "output": blip_out,
            "inference_time_sec": instructblip_infer_time
        })

    # --- Save results to separate CSVs ---
    idefics_csv = os.path.join(out_dir, "idefics_results.csv")
    instructblip_csv = os.path.join(out_dir, "instructblip_results.csv")

    pd.DataFrame(idefics_results).to_csv(idefics_csv, index=False, quoting=1)
    pd.DataFrame(instructblip_results).to_csv(instructblip_csv, index=False, quoting=1)

    print(f"\nIDEFICS results saved to {idefics_csv}")
    print(f"InstructBLIP results saved to {instructblip_csv}")

    # --- Print model load times ---
    print(f"\nModel load times:")
    print(f"IDEFICS: {idefics_load_time:.2f} sec")
    print(f"InstructBLIP: {instructblip_load_time:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True, help="Path to images folder")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of images")
    args = parser.parse_args()

    run_inference(args.img_root, args.out_dir, args.limit)


