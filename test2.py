import cv2
import numpy as np
from PIL import Image

def calculate_color_difference(img1, img2):
    """Calculate the Euclidean color difference between two images."""
    return np.sqrt(np.sum((img1.astype("float") - img2.astype("float")) ** 2, axis=2))

def visualize_color_difference(image1_path, image2_path, output_dir):
    """Create and save a color map showing differences between two images."""
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Ensure images are the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Calculate color difference
    color_diff = calculate_color_difference(img1, img2)

    # Normalize differences to the range 0-255
    normalized_diff = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a full-color map (e.g., COLORMAP_JET) to the normalized differences
    color_map = cv2.applyColorMap(normalized_diff, cv2.COLORMAP_JET)

    # Calculate total score based on the sum of all differences
    total_score = int(np.sum(color_diff))

    # Save the result
    output_path = f"{output_dir}/{total_score}.png"
    cv2.imwrite(output_path, color_map)
    print(f"Color difference map saved as {output_path}")

# Example usage
visualize_color_difference("image.jpg", "filled.jpg", "folder")
