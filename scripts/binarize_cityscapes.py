import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# a label and all meta information
class Label(NamedTuple):
    # The identifier of this label, e.g. 'car', 'person', ...
    name: str

    # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.
    id: int

    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!
    trainId: int

    # The name of the category that this label belongs to
    category: str

    # The ID of this category. Used to create ground truth images on category level.
    categoryId: int

    # Whether this label distinguishes between single instances or not
    hasInstances: bool

    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not
    ignoreInEval: bool

    # The RGB color of this label
    color: tuple[int, int, int]


def process_common_split(
    dump_path: Path,
    cityscapes_path: Path,
    used_labels: Sequence[Label],
    split: Literal["train", "val"],
):
    split_long = "validation" if split == "val" else "training"

    images_dump_path = dump_path / "images" / split_long
    images_dump_path.mkdir(parents=True, exist_ok=True)

    annotations_dump_path = dump_path / "annotations" / split_long
    annotations_dump_path.mkdir(parents=True, exist_ok=True)

    mask_files = tuple(cityscapes_path.glob(f"gtFine/{split}/**/*gtFine_labelIds.png"))

    with tqdm(used_labels) as classes:
        for cls in classes:
            cls_annotations_dump_path = annotations_dump_path / cls.name
            cls_annotations_dump_path.mkdir(exist_ok=True)

            for mask_idx, mask_file in enumerate(mask_files, 1):
                classes.set_description(
                    f"For {split_long} -- class: '{cls.name}', "
                    f"scanned {mask_idx}/{len(mask_files)}"
                )

                with Image.open(mask_file) as im:
                    mask = np.asarray(im)

                if cls.id not in mask:
                    continue

                mask = (mask == cls.id).astype(np.uint8) * 255
                mask = Image.fromarray(mask, mode="L")

                city_name = mask_file.parent.name
                mask_new_path = cls_annotations_dump_path / mask_file.name.replace(
                    "_gtFine_labelIds", ""
                )
                mask.save(mask_new_path, optimize=True)

                image_old_path = (
                    cityscapes_path
                    / "leftImg8bit"
                    / split
                    / city_name
                    / f"{mask_file.stem.replace('gtFine_labelIds', 'leftImg8bit')}.png"
                )
                image_new_path = (
                    images_dump_path
                    / f"{image_old_path.stem.replace('_leftImg8bit', '')}.png"
                )

                # If the image file already exists skip the copy part below
                if not image_new_path.exists():
                    shutil.copy(image_old_path, image_new_path)


def main(cityscapes_path: Path, dump_path: Path):
    """Prepare the cityscapes dataset compatible for few shot training.
        <dump_path>
        |---images
        |   |---training
        |         |---ADE_train_00001030.png
        |         |---............
        |   |   .......
        |   |---validation
        |           |---ADE_train_00001030.png
        |           |---............
        |       ......
        |---annotations
        |   |
        |   |---training
        |   |   |---<class_name> For. eg. car|human
        |   |   |.......(similar to above)

    Args:
        K: The number of data samples in K-shot learners.
        cityscapes_path: The path of the cityscape original dataset for
    """

    # --------------------------------------------------------------------------------
    # A list of all labels
    # --------------------------------------------------------------------------------

    # Please adapt the train IDs as appropriate for your approach.
    # Note that you might want to ignore labels with ID 255 during training.
    # Further note that the current train IDs are only a suggestion. You can use whatever you like.
    # Make sure to provide your results using the original IDs and not the training IDs.
    # Note that many IDs are ignored in evaluation and thus you never need to predict these!

    labels = (
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    )

    # Use only the labels not ignore in the eval split (19 classes)
    used_labels = tuple(label for label in labels if not label.ignoreInEval)

    common_args = (dump_path, cityscapes_path, used_labels)

    # Prepare train dataset

    process_common_split(*common_args, split="train")

    # Prepare validation dataset
    process_common_split(*common_args, split="val")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="CityScapes Parser",
        description=(
            "The arguments needed for the dataset preparation are "
            "passed within this parser."
        ),
    )

    parser.add_argument(
        "--cityscapes_path",
        type=Path,
        required=True,
        help="The path of Cityscapes dataset",
    )
    parser.add_argument(
        "--dump_path",
        type=Path,
        help="The path to save new dataset",
        required=True,
    )

    args = parser.parse_args()

    main(**vars(args))
