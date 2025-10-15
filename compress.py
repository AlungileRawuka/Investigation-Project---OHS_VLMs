import os
from PIL import Image
from tqdm import tqdm

# Path to your main images folder
base_dir = r"C:/Users/Alungile/YOS4/Investigation Project/Investigation-Project---OHS_VLMs/data/images"

# Output folder
output_base = os.path.join(base_dir, "compressed")

# Compression settings
output_quality = 70  # JPEG quality (0-100)
max_size = (1024, 1024)  # Max width/height

# Supported image formats
image_extensions = ('.jpg', '.jpeg', '.png')

# Walk through all subfolders
for root, dirs, files in os.walk(base_dir):
    # Skip the output folder itself
    if root.startswith(output_base):
        continue

    # Compute the corresponding output folder
    relative_path = os.path.relpath(root, base_dir)
    output_folder = os.path.join(output_base, relative_path)
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(files, desc=f"Processing {relative_path}"):
        if file.lower().endswith(image_extensions):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".jpg")  # Save all as JPEG

            try:
                with Image.open(input_path) as img:
                    # Convert to RGB if needed
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")

                    # Resize while keeping aspect ratio
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                    # Save compressed image
                    img.save(output_path, optimize=True, quality=output_quality)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

print("All images compressed and saved in 'compressed/' folders!")

