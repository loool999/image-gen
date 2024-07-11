import operator
from PIL import Image, ImageChops, ImageEnhance
import random
import numpy
import copy
import os
import shutil
import cv2
from tqdm import tqdm

NUMBER_OF_STARTING_OBJECTS = 300
SURVIVORS = 1/4
OLD_AGE = False
CHILDREN_COUNT = 3
GENERATIONS = 50
MUTATION_RATE = 0.93
MUTATION_RATE_COLOR = 0.98
OBJECTS_COUNT = 100

# Directories
NEW_OBJECTS_DIR = "new_object"
SINGLE_DIR = "single"
DONE_DIR = "done"

# Create directories if they don't exist
os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)

# Load images from new_objects folder
new_objects_images = [Image.open(os.path.join(NEW_OBJECTS_DIR, file)).convert("RGBA") for file in os.listdir(NEW_OBJECTS_DIR) if file.endswith(('png', 'jpg', 'jpeg'))]

# Open the goal image
image = Image.open("image.jpg").convert("RGBA")

# Make empty canvas with size of the image
canvas = Image.new("RGBA", image.size)

# Define object class
class Object:
    def __init__(self):
        self.image = random.choice(new_objects_images)
        self.size = self.image.size
        self.coordinates = [random.randint(0, canvas.width), random.randint(0, canvas.height)]
        self.angle = random.randint(0, 359)
        self.color = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
            #Function where the object gets drawn on canvas
    def draw(self,canvas):
        #Make Image of Object, we will put object onto that image and then we can rotate it
        object = Image.new("RGBA",self.size,tuple(self.color))
        object = object.rotate(self.angle,expand=True)

        #calculate the coordinates since the coordinates i set are supposed to be for center of the object
        x, y = self.coordinates[0] - (object.size[0] // 2), self.coordinates[1] - (object.size[1] // 2)

        #Draw the Object
        canvas.paste(object, (x,y),object)
    '''
    def draw(self, canvas):
        object_img = self.image.rotate(self.angle, expand=True)
        overlay = Image.new("RGBA", object_img.size, tuple(self.color + [128]))  # 128 for 50% opacity
        object_img = ImageChops.blend(object_img, overlay, 0.5)
        x, y = self.coordinates[0] - (object_img.size[0] // 2), self.coordinates[1] - (object_img.size[1] // 2)
        canvas.paste(object_img, (x, y), object_img)
    '''
    def reproduce(self, hospital):
        for _ in range(CHILDREN_COUNT):
            child = copy.deepcopy(self)
            x_range = range(-self.coordinates[0], image.width - self.coordinates[0])
            y_range = range(-self.coordinates[1], image.height - self.coordinates[1])
            child.coordinates[0] += random.choices(x_range, weights=[MUTATION_RATE ** abs(x) for x in x_range], k=1)[0]
            child.coordinates[1] += random.choices(y_range, weights=[MUTATION_RATE ** abs(x) for x in y_range], k=1)[0]

            #Mutate size of the Object
            size_x_range = range(-self.size[0]+1,image.width-self.size[0])
            size_y_range = range(-self.size[1]+1,image.height-self.size[1])
            child.size = list(self.size)
            child.size[0] += random.choices(size_x_range, weights=[MUTATION_RATE ** abs(x) for x in size_x_range], k=1)[0]
            child.size[1] += random.choices(size_y_range, weights=[MUTATION_RATE ** abs(x) for x in size_y_range], k=1)[0]
            self.size = tuple(child.size)
            angle_range = range(-360, 360)
            child.angle += random.choices(angle_range, weights=[MUTATION_RATE ** abs(x) for x in angle_range], k=1)[0]

            red_range = range(-self.color[0], 255 - self.color[0])
            green_range = range(-self.color[1], 255 - self.color[1])
            blue_range = range(-self.color[2], 255 - self.color[2])

            child.color[0] += random.choices(red_range, weights=[MUTATION_RATE_COLOR ** abs(x) for x in red_range], k=1)[0]
            child.color[1] += random.choices(green_range, weights=[MUTATION_RATE_COLOR ** abs(x) for x in green_range], k=1)[0]
            child.color[2] += random.choices(blue_range, weights=[MUTATION_RATE_COLOR ** abs(x) for x in blue_range], k=1)[0]

            hospital.append(child)

generation_count = 0

for obj_idx in tqdm(range(OBJECTS_COUNT), desc="Processing Objects"):
    objects = [Object() for _ in range(NUMBER_OF_STARTING_OBJECTS)]

    for generation in tqdm(range(GENERATIONS), desc=f"Generations for Object {obj_idx+1}"):
        generation_count += 1
        for object in objects:
            canvas_copy = canvas.copy()
            object.draw(canvas_copy)
            difference_with_object = numpy.sum(numpy.abs(numpy.array(canvas_copy).astype(numpy.int16) - numpy.array(image).astype(numpy.int16)))
            difference_without_object = numpy.sum(numpy.abs(numpy.array(canvas).astype(numpy.int16) - numpy.array(image).astype(numpy.int16)))
            score = difference_without_object - difference_with_object
            object.score = score

        objects.sort(reverse=True, key=operator.attrgetter('score'))
        objects = objects[:int(len(objects) * SURVIVORS)]

        if generation + 1 != GENERATIONS:
            children = []
            for survivor in objects:
                survivor.reproduce(children)
            if OLD_AGE:
                objects = children
            else:
                objects = children + objects

    best_object = objects[0]
    best_object.draw(canvas)
    canvas.save(f"{SINGLE_DIR}/object_{obj_idx+1}.png")

canvas.save(f"{DONE_DIR}/canvas.png")
