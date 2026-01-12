import os

def rename_images_in_folder(folder_path):
    # List all files and filter for image files
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # Sort files alphabetically
    for idx, filename in enumerate(files, 1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{idx}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
    print(f"Renamed {len(files)} files in {folder_path}")

if __name__ == "__main__":
    base_path = os.path.join("..", "data", "paired", "deblur")
    for subfolder in ["input", "target"]:
        folder = os.path.join(base_path, subfolder)
        rename_images_in_folder(folder)
