# image_processing.py
from PIL import Image, ImageEnhance
from collections import Counter
import cv2
import numpy as np
import os
import random

# Constants
NUMBER_OF_STARTING_OBJECTS = 100
SURVIVORS = 1/4
OLD_AGE = False
CHILDREN_COUNT = 3
GENERATIONS = 20
MUTATION_RATE = 0.93
MUTATION_RATE_COLOR = 0.98
OBJECTS_COUNT = 50

input_image_path = 'image.jpg'
output_image_path = 'filled.jpg'
object_folder = 'new_object'
process_folder = 'process'
phase_folder = 'phases'

# Ensure directories exist
if not os.path.exists(process_folder):
    os.makedirs(process_folder)
if not os.path.exists(phase_folder):
    os.makedirs(phase_folder)

def apply_tint(image, tint_color, blend_factor=0.5):
    tint_layer = Image.new("RGBA", image.size, tint_color)
    tinted_image = Image.blend(image, tint_layer, blend_factor)
    alpha = image.split()[3]
    tinted_image.putalpha(alpha)
    return tinted_image

def get_most_frequent_color(image, region):
    region_crop = image.crop(region)
    pixels = list(region_crop.getdata())
    pixel_counts = Counter(pixels)
    most_frequent_pixel = pixel_counts.most_common(1)[0][0]
    return most_frequent_pixel

def mutate_image(image, mutation_rate, mutation_rate_color):
    pixels = np.array(image)
    mutation_mask = np.random.rand(*pixels.shape[:2]) < mutation_rate
    color_mutation_mask = np.random.rand(*pixels.shape[:2]) < mutation_rate_color
    
    pixels[mutation_mask] = np.random.randint(0, 256, size=(np.count_nonzero(mutation_mask), pixels.shape[2]))
    
    for i in range(3):
        channel_mutation_mask = color_mutation_mask & (np.random.rand(*pixels.shape[:2]) < mutation_rate_color)
        pixels[channel_mutation_mask, i] = np.random.randint(0, 256, size=np.count_nonzero(channel_mutation_mask))

    mutated_image = Image.fromarray(pixels)
    return mutated_image

def paste_object_image(input_image_path, base_image, object_folder, output_v1_image_path, phase, num_phases):
    base_img = base_image.copy()
    object_image_path = random.choice([os.path.join(object_folder, f) for f in os.listdir(object_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    with Image.open(object_image_path) as obj_img:
        obj_img = obj_img.convert('RGBA')
        
        max_scale = 15
        min_scale = 0.05
        scale_factor = max_scale - ((max_scale - min_scale) * (phase / num_phases))
        new_size = (int(obj_img.width * scale_factor), int(obj_img.height * scale_factor))
        obj_img = obj_img.resize(new_size, Image.LANCZOS)
        
        angle = random.uniform(0, 360)
        obj_img = obj_img.rotate(angle, expand=True)
        
        max_x = base_img.width - obj_img.width
        max_y = base_img.height - obj_img.height
        if max_x < 0 or max_y < 0:
            max_x = max(0, base_img.width - obj_img.width + int(0.2 * obj_img.width))
            max_y = max(0, base_img.height - obj_img.height + int(0.2 * obj_img.height))
        paste_position = (
            random.randint(max(-int(0.2 * obj_img.width), -max_x), max(0, max_x)),
            random.randint(max(-int(0.2 * obj_img.height), -max_y), max(0, max_y))
        )
        
        with Image.open(input_image_path) as input_img:
            input_img = input_img.convert('RGBA')
            region = (max(0, paste_position[0]), max(0, paste_position[1]), min(base_img.width, paste_position[0] + obj_img.width), min(base_img.height, paste_position[1] + obj_img.height))
            tint_color = get_most_frequent_color(input_img, region)
            tint_color = (*tint_color[:3], 255)  # Ensure full opacity

        obj_img = apply_tint(obj_img, tint_color, blend_factor=0.5)
        base_img.paste(obj_img, paste_position, obj_img)
        base_img.save(output_v1_image_path)

def calculate_similarity(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    similarity_percentage = min(100, 100 * (psnr / 48))
    return similarity_percentage

def create_colormap():
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        colormap[i, 0, 2] = int(255 * (i / 255.0))
        colormap[i, 0, 1] = int(255 * ((255 - i) / 255.0))
    return colormap

def visualize_color_difference(image1_path, image2_path, output_dir):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    similarity_score = calculate_similarity(img1, img2)
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    colormap = create_colormap()
    diff_color = cv2.applyColorMap(diff_gray, colormap)
    result = cv2.addWeighted(img1, 0.7, diff_color, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Similarity: {similarity_score:.2f}%", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'difference_map.png'), diff_color)
    return similarity_score

def process_single_iteration(input_image_path, base_image, object_folder, process_folder, phase_folder, phase, iteration, num_phases):
    output_v1_image_path = os.path.join(process_folder, f'output-v1-{phase}-{iteration}.png')
    paste_object_image(input_image_path, base_image, object_folder, output_v1_image_path, phase, num_phases)
    score = visualize_color_difference(input_image_path, output_v1_image_path, process_folder)
    return (score, output_v1_image_path)

def evolve_images(input_image_path, object_folder, generations, mutation_rate, mutation_rate_color):
    base_image = Image.open(input_image_path).convert("RGBA")
    population = [base_image] * NUMBER_OF_STARTING_OBJECTS
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        scores = []
        for idx, base_image in enumerate(population):
            score, output_v1_image_path = process_single_iteration(
                input_image_path,
                base_image,
                object_folder,
                process_folder,
                phase_folder,
                generation,
                idx,
                generations
            )
            scores.append((score, output_v1_image_path))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        survivors = scores[:int(len(scores) * SURVIVORS)]
        population = []
        
        for score, img_path in survivors:
            img = Image.open(img_path).convert("RGBA")
            population.append(img)
            for _ in range(CHILDREN_COUNT):
                child_img = mutate_image(img, mutation_rate, mutation_rate_color)
                population.append(child_img)
        
        print(f"Best score in generation {generation + 1}: {scores[0][0]:.2f}%")

    best_image_path = scores[0][1]
    final_image = Image.open(best_image_path).convert("RGBA")
    final_image.save(output_image_path)
    print(f"Best image saved to {output_image_path} with score: {scores[0][0]:.2f}%")

evolve_images(input_image_path, object_folder, GENERATIONS, MUTATION_RATE, MUTATION_RATE_COLOR)
