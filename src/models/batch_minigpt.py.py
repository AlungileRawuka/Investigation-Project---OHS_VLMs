import os
import csv
from test import *  # your OFA wrapper

# Folder with images
folder_path = "images"

# Output CSV file
csv_path = "ohs_results.csv"

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
                description = main(file_path, prompt="describe and list all the occupational health and safety issues in the image in a concise manner")
                
                # Write to CSV
                writer.writerow(["MiniGPT-4", filename, description])
                print(f"Processed {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
