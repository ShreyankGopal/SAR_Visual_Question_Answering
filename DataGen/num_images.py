import os

def count_images(directory):
    image_extensions = {'.tif'}
    count = 0
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            count += 1
    return count

directory = "/home/saishruti/Research1/Datasets/OpenEarthMap/train/sar_images"
num_images = count_images(directory)
print(f"Number of images in '{directory}': {num_images}")
