from PIL import Image
from collections import Counter

def create_most_frequent_pixel_image(input_image_path, output_image_path):
    # Open the input image
    with Image.open(input_image_path) as img:
        # Convert the image to RGB mode
        img = img.convert('RGB')
        # Get all pixels as a list of tuples
        pixels = list(img.getdata())
        
        # Count the frequency of each pixel
        pixel_counts = Counter(pixels)
        
        # Find the most frequent pixel
        most_frequent_pixel = pixel_counts.most_common(1)[0][0]
        
        # Create a new image filled with the most frequent pixel
        new_img = Image.new('RGB', img.size, most_frequent_pixel)
        
        # Save the new image
        new_img.save(output_image_path)

# Example usage
input_image_path = 'image.jpg'
output_image_path = 'output.jpg'
create_most_frequent_pixel_image(input_image_path, output_image_path)
