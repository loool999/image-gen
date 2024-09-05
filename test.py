import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_image(path):
    return Image.open(path).convert("RGB")

def compute_color_difference(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    if arr1.shape != arr2.shape:
        raise ValueError("Images must be of the same size")
    
    diff = np.sqrt(np.sum((arr1 - arr2) ** 2, axis=-1))
    return diff

def generate_heatmap(diff):
    norm_diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))  # Normalize to [0, 1]
    
    # Define the color map
    colors = [(0, 0, 1), (1, 0, 0)]  # Light blue to red
    cmap = LinearSegmentedColormap.from_list("custom_heatmap", colors, N=256)
    
    heatmap = cmap(norm_diff)
    return heatmap

def save_heatmap(heatmap, path):
    plt.imsave(path, heatmap)

def calculate_total_difference_score(diff):
    return int(np.sum(diff))

def main(image_path1, image_path2, heatmap_path):
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)
    
    diff = compute_color_difference(img1, img2)
    heatmap = generate_heatmap(diff)
    
    save_heatmap(heatmap, heatmap_path)
    
    total_score = calculate_total_difference_score(diff)
    print(f"Total Difference Score: {total_score}")

# Usage
image_path1 = "image.jpg"
image_path2 = "filled.jpg"
heatmap_path = "heatmap_output.png"

main(image_path1, image_path2, heatmap_path)
