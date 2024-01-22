from pathlib import Path
from PIL import Image
import numpy as np

from tqdm import tqdm


DATA_ROOT = "/mnt/SSD1/rabin/datasets/phrasecut/images"

shapes = []
for img_path in tqdm(tuple(Path(DATA_ROOT).glob("*.jpg")), desc="Loading Images"):
    with Image.open(img_path) as img:
        shapes.append(img.size)

shape_array = np.asarray(shapes)

print("Array Shape: ", shape_array.shape)

print("Min: ", shape_array.min(0))
print("Max: ", shape_array.max(0))
print("Mean: ", shape_array.mean(0))
print("Std: ", shape_array.std(0))
