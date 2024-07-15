import operator
from PIL import Image, ImageChops, ImageEnhance
import random
import numpy as np
import copy
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

NUMBER_OF_STARTING_OBJECTS = 300
SURVIVORS = 1 / 4
OLD_AGE = False
CHILDREN_COUNT = 5
GENERATIONS = 10
MUTATION_RATE = 0.93
MUTATION_RATE_COLOR = 0.98
OBJECTS_COUNT = 100

NEW_OBJECTS_DIR = "new_object"
SINGLE_DIR = "single"
DONE_DIR = "done"

os.makedirs(SINGLE_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)

new_objects_images = [Image.open(os.path.join(NEW_OBJECTS_DIR, file)).convert("RGBA") for file in os.listdir(NEW_OBJECTS_DIR) if file.endswith(('png', 'jpg', 'jpeg'))]
image = Image.open("image.jpg").convert("RGBA")


class Object:
    def __init__(self):
        self.image = random.choice(new_objects_images)
        self.size = self.image.size
        self.coordinates = [random.randint(0, image.width), random.randint(0, image.height)]
        self.angle = random.randint(0, 359)
        self.color = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]

    def draw(self, canvas):
        object_img = self.image.rotate(self.angle, expand=True)
        overlay = Image.new("RGBA", object_img.size, tuple(self.color + [128]))  # 128 for 50% opacity
        object_img = ImageChops.blend(object_img, overlay, 0.5)
        x, y = self.coordinates[0] - (object_img.size[0] // 2), self.coordinates[1] - (object_img.size[1] // 2)
        canvas.paste(object_img, (x, y), object_img)

    def reproduce(self, hospital):
        for _ in range(CHILDREN_COUNT):
            child = copy.deepcopy(self)
            x_range = range(-self.coordinates[0], image.width - self.coordinates[0])
            y_range = range(-self.coordinates[1], image.height - self.coordinates[1])
            child.coordinates[0] += random.choices(x_range, weights=[MUTATION_RATE ** abs(x) for x in x_range], k=1)[0]
            child.coordinates[1] += random.choices(y_range, weights=[MUTATION_RATE ** abs(y) for y in y_range], k=1)[0]

            size_x_range = range(-self.size[0] + 1, image.width - self.size[0])
            size_y_range = range(-self.size[1] + 1, image.height - self.size[1])
            child.size = list(self.size)
            child.size[0] += random.choices(size_x_range, weights=[MUTATION_RATE ** abs(x) for x in size_x_range], k=1)[0]
            child.size[1] += random.choices(size_y_range, weights=[MUTATION_RATE ** abs(y) for y in size_y_range], k=1)[0]
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


def process_single_object(obj_idx, new_objects_images, image, canvas_size):
    objects = [Object() for _ in range(NUMBER_OF_STARTING_OBJECTS)]
    canvas = Image.new("RGBA", canvas_size)

    for generation in range(GENERATIONS):
        for obj in objects:
            obj.draw(canvas)
            difference_with_object = np.sum(np.abs(np.array(canvas).astype(np.int16) - np.array(image).astype(np.int16)))
            difference_without_object = np.sum(np.abs(np.array(canvas).astype(np.int16) - np.array(image).astype(np.int16)))
            obj.score = difference_without_object - difference_with_object

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
    canvas.save(f"{SINGLE_DIR}/object_{obj_idx + 1}.png")
    return f"{SINGLE_DIR}/object_{obj_idx + 1}.png"


if __name__ == "__main__":
    canvas_size = image.size
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_object, idx, new_objects_images, image, canvas_size) for idx in range(OBJECTS_COUNT)]
        results = [future.result() for future in tqdm(futures, total=OBJECTS_COUNT, desc="Processing Objects")]

    final_canvas = Image.new("RGBA", canvas_size)
    for result_path in results:
        obj_image = Image.open(result_path).convert("RGBA")
        final_canvas.paste(obj_image, (0, 0), obj_image)
    final_canvas.save(f"{DONE_DIR}/canvas.png")
