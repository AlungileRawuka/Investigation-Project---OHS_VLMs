import os
import csv
from ofa import *  # your OFA wrapper

# Folder with images
folder_path = "images"

# Output CSV file
csv_path = "ohs_results.csv"

model = OFAWrapper()

# Create/open CSV file
with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Model","filename", "description"])  # header

    # Iterate all image files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Run OFA model
                description = model.generate_caption(file_path)
                
                # Write to CSV
                writer.writerow(["OFA", filename, description])
                print(f"Processed {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

