import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

NUMBER_OF_STARTING_OBJECTS = 40
SURVIVORS = 40
OLD_AGE = False
CHILDREN_COUNT = 40
GENERATIONS = 30
MUTATION_RATE = 0.93
MUTATION_RATE_COLOR = 0.98
MUTATION_RATE_SIZE = 0.93
OBJECTS_COUNT = 2000

# Directories
NEW_OBJECTS_DIR = "object"
SINGLE_DIR = "single"
DONE_DIR = "done"

# Create directories if they don't exist
os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)

# Load images from new_objects folder
new_objects_images = [cv2.imread(os.path.join(NEW_OBJECTS_DIR, file), cv2.IMREAD_UNCHANGED) for file in os.listdir(NEW_OBJECTS_DIR) if file.endswith(('png', 'jpg', 'jpeg'))]

# Open the goal image
image = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)

# Make empty canvas with size of the image
canvas = np.zeros(image.shape, dtype=np.uint8)


# Define object class
class Object:
    def __init__(self):
        self.image = new_objects_images[np.random.randint(0, len(new_objects_images))]
        self.size = self.image.shape[:2]
        self.coordinates = np.random.randint(0, canvas.shape[1], size=2)
        self.angle = np.random.randint(0, 360)
        self.color = np.random.randint(0, 256, size=3)

    def draw(self, canvas):
        rot_mat = cv2.getRotationMatrix2D((self.size[1] / 2, self.size[0] / 2), self.angle, 1.0)
        rotated_image = cv2.warpAffine(self.image, rot_mat, self.size)
        mask = rotated_image[:, :, 3] / 255.0
        inv_mask = 1.0 - mask
        x, y = self.coordinates[0] - (self.size[1] // 2), self.coordinates[1] - (self.size[0] // 2)
        canvas[y:y+self.size[0], x:x+self.size[1]] = (mask[:, :, np.newaxis] * self.color.reshape(1, 1, 3) + inv_mask[:, :, np.newaxis] * canvas[y:y+self.size[0], x:x+self.size[1]].transpose(1, 0, 2)).transpose(1, 0, 2)

    def reproduce(self, hospital):
        for _ in range(CHILDREN_COUNT):
            child = Object()
            child.image = self.image.copy()
            child.size = self.size
            child.coordinates = self.coordinates + np.random.choice(range(-self.coordinates[0], image.shape[1] - self.coordinates[0]), size=2, p=[MUTATION_RATE ** abs(x) for x in range(-self.coordinates[0], image.shape[1] - self.coordinates[0])])
            if np.random.random() < MUTATION_RATE_SIZE:
                new_size = self.size + np.random.choice(range(-self.size[0]+1, image.shape[1]-self.size[0]), size=2, p=[MUTATION_RATE_SIZE ** abs(x) for x in range(-self.size[0]+1, image.shape[1]-self.size[0])])
                new_size = np.maximum(new_size, 1)
                child.image = cv2.resize(child.image, tuple(new_size[::-1]))
                child.size = new_size
            if np.random.random() < MUTATION_RATE:
                child.angle += np.random.choice(range(-360, 360), p=[MUTATION_RATE ** abs(x) for x in range(-360, 360)])
            if np.random.random() < MUTATION_RATE_COLOR:
                child.color += np.random.choice(range(-self.color[0], 255 - self.color[0]), size=3, p=[MUTATION_RATE_COLOR ** abs(x) for x in range(-self.color[0], 255 - self.color[0])])
            hospital.append(child)

def process_object(obj_idx):
    objects = [Object() for _ in range(NUMBER_OF_STARTING_OBJECTS)]
    for generation in range(GENERATIONS):
        canvas_copy = canvas.copy()
        scores = Parallel(n_jobs=-1)(delayed(calculate_score)(object, canvas_copy, image) for object in objects)
        objects = [obj for _, obj in sorted(zip(scores, objects), reverse=True)]
        objects = objects[:int(len(objects) * SURVIVORS)]
        children = []
        for survivor in objects:
            survivor.reproduce(children)
        if OLD_AGE:
            objects = children
        else:
            objects = children + objects
    best_object = objects[0]
    best_object.draw(canvas)
    cv2.imwrite(f"{SINGLE_DIR}/object_{obj_idx+1}.png", canvas)
    
def calculate_score(object, canvas, image):
    canvas_copy = np.zeros(canvas.shape, dtype=np.uint8)
    object.draw(canvas_copy)
    difference_with_object = np.sum(np.abs(canvas_copy.astype(np.int16) - image.astype(np.int16)))
    difference_without_object = np.sum(np.abs(canvas.astype(np.int16) - image.astype(np.int16)))
    score = difference_without_object - difference_with_object
    return score

Parallel(n_jobs=-1)(delayed(process_object)(obj_idx) for obj_idx in tqdm(range(OBJECTS_COUNT), desc="Processing Objects"))

cv2.imwrite(f"{DONE_DIR}/canvas.png", canvas)
