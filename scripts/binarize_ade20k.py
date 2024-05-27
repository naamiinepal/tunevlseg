import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

ADE_classes = (
    "background",
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed ",
    "windowpane, window",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
)


def prepare_common_split(
    ADE20K_path: Path, dump_path: Path, split: Literal["training", "validation"]
):
    images_dump_path = dump_path / "images" / split
    images_dump_path.mkdir(parents=True, exist_ok=True)

    annotations_dump_path = dump_path / "annotations" / split
    annotations_dump_path.mkdir(parents=True, exist_ok=True)

    start_idx = 1
    # Skip background class
    useful_classes = ADE_classes[start_idx:]
    with tqdm(useful_classes, initial=start_idx) as classes:
        for idx, cls in enumerate(classes, start_idx):
            mask_files = tuple(ADE20K_path.glob(f"annotations/{split}/*.png"))

            cls_annotations_dump_path = annotations_dump_path / cls
            cls_annotations_dump_path.mkdir(exist_ok=True)

            for mask_idx, mask_file in enumerate(mask_files, 1):
                classes.set_description(
                    f"For {split} -- class: '{cls}', "
                    f"scanned {mask_idx}/{len(mask_files)}"
                )

                with Image.open(mask_file) as img:
                    mask = np.asarray(img)

                if idx not in mask:
                    continue

                mask = (mask == idx).astype(np.uint8) * 255
                mask = Image.fromarray(mask, mode="L")

                mask_new_path = cls_annotations_dump_path / mask_file.name
                mask.save(mask_new_path, optimize=True)

                image_basename = f"{mask_file.stem}.jpg"

                image_old_path = ADE20K_path / "images" / split / image_basename

                image_new_path = images_dump_path / image_basename

                # If the image file already exists skip the copy part below
                if not image_new_path.exists():
                    shutil.copy(image_old_path, image_new_path)


def main(ADE20K_path: Path, dump_path: Path):
    """Prepare the ADE20K dataset compatible for few shot training.
        If shot (K) is 16, a folder sibling to ADE20K_path  is created with name '16_shot', having the structure:
            <dump_path>
            |---images
            |   |---training
            |   |   |---airplane
            |   |      |---ADE_train_00001030.jpg
            |   |       |---............
            |   |   .......
            |   |---validation
            |       |---airplane
            |           |---ADE_train_00001030.png
            |           |---............
            |       ......
            |---annotations
                |---.......(similar to above)

    Args:
        K: The number of data samples in K-shot learners.
        ADE20K_path: The path of the ADEChallengeData2016 dataset for
    """

    common_args = (ADE20K_path, dump_path)

    # Prepare train dataset
    prepare_common_split(*common_args, split="training")

    # Prepare validation dataset
    prepare_common_split(*common_args, split="validation")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="ADE20K Parser",
        description=(
            "The arguments needed for the dataset preparation are "
            "passed within this parser."
        ),
    )

    parser.add_argument(
        "--ade20k_path",
        type=Path,
        help="The path of ADE20K dataset",
        required=True,
    )

    parser.add_argument(
        "--dump_path",
        type=Path,
        help="The path to save new dataset",
        required=True,
    )

    args = parser.parse_args()

    main(**vars(args))
