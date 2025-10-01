import os
import argparse
import pandas as pd
from tabulate import tabulate
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

    # Initialize models (CPU only for stability)
    idefics = IDEFICSWrapper()
    instructblip = InstructBLIPWrapper()

    results = []
    for img_file in selected_images:
        img_path = os.path.join(img_root, img_file)
        print(f"Processing {img_file}...")

        try:
            idefics_out = idefics.run(img_path)
        except Exception as e:
            idefics_out = f"Error: {str(e)}"

        try:
            blip_out = instructblip.run(img_path)
        except Exception as e:
            blip_out = f"Error: {str(e)}"

        results.append({
            "image": img_file,
            "idefics_output": idefics_out,
            "instructblip_output": blip_out,
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, "results.csv")
    df.to_csv(out_path, index=False, quoting=1)
    print(f"Saved results to {out_path}")

    # Save as table
    table_str = tabulate(
    results,
    headers="keys",  # use dict keys, works with list of dicts
    tablefmt="grid",
    maxcolwidths=[45, 50, 50],
    showindex=False
    )

    out_table_txt = os.path.join(out_dir, "results_table.txt")
    with open(out_table_txt, "w", encoding="utf-8") as f:
        f.write(table_str)

    print(f"\nTable saved to {out_table_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True, help="Path to images folder")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of images")
    args = parser.parse_args()

    run_inference(args.img_root, args.out_dir, args.limit)


