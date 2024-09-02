import concurrent.futures
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

class_names = np.array(
    (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
)

color_maps = np.array(
    (
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
    ),
    dtype=np.uint8,
)[:, None, None, :]

if len(color_maps) != len(class_names):
    raise ValueError(
        f"The number of color_maps: {len(color_maps)} and class_names: {len(class_names)} should be same"
    )


def convert_to_binary_mask(mask: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    segmentation_masks = np.all(mask == color_maps, axis=-1)
    non_empty_indices = np.any(segmentation_masks, axis=(1, 2))

    selected_masks = segmentation_masks[non_empty_indices].astype(np.uint8) * 255
    selected_class_names = tuple(class_names[non_empty_indices])

    return selected_class_names, selected_masks


def process_single_mask(
    mask_dir: Path, mask_output_dir: Path, image_id: Any, dryrun: bool
):
    mask_path = mask_dir / f"{image_id}.png"
    mask = cv2.imread(str(mask_path))

    if mask is None:
        raise ValueError(f"Could not find mask for image {image_id}")

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    current_mask_names, segmentation_masks = convert_to_binary_mask(mask)

    for mask_name, binary_mask in zip(
        current_mask_names, segmentation_masks, strict=True
    ):
        save_path = mask_output_dir / f"{mask_name}/{image_id}.png"
        if not dryrun and not save_path.is_file():
            ok = cv2.imwrite(
                str(save_path),
                binary_mask,
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )

            if not ok:
                raise ValueError(
                    f"Could not save mask {mask_name} for image {image_id}"
                )

    return current_mask_names


StrOrPath = str | Path


def binarize_masks(
    mask_dir: Path,
    mask_output_root: Path,
    train_image_ids: Iterable[Any],
    val_image_ids: Iterable[Any],
    max_workers: int | None,
    dryrun: bool,
    verbose: bool,
) -> None:
    train_mask_output_dir = mask_output_root / "training"
    val_mask_output_dir = mask_output_root / "validation"

    if not dryrun:
        train_mask_output_dir.mkdir(parents=True, exist_ok=True)
        val_mask_output_dir.mkdir(exist_ok=True)

        for cname in class_names:
            train_mask_output_dir.joinpath(cname).mkdir(exist_ok=True)
            val_mask_output_dir.joinpath(cname).mkdir(exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
        # Submit the training image ids
        future_to_image_id = {
            executor.submit(
                process_single_mask, mask_dir, train_mask_output_dir, image_id, dryrun
            ): image_id
            for image_id in train_image_ids
        }

        # Now submit the validation image ids
        for image_id in val_image_ids:
            future_to_image_id[
                executor.submit(
                    process_single_mask,
                    mask_dir,
                    val_mask_output_dir,
                    image_id,
                    dryrun,
                )
            ] = image_id

        for future in tqdm(
            concurrent.futures.as_completed(future_to_image_id),
            desc="Saving train and validation masks",
            total=len(future_to_image_id),
        ):
            image_id = future_to_image_id[future]
            try:
                save_path = future.result()
            except (
                FileNotFoundError,
                cv2.error,
                UnicodeError,
                ValueError,
            ) as exc:
                print(f"{image_id} generated an exception: {exc}")
            else:
                if verbose:
                    print(f"{image_id} saved to {save_path}")


def main(
    voc_root: StrOrPath,
    output_root: StrOrPath,
    max_workers: int | None,
    skip_mask_save: bool = False,
    skip_image_save: bool = False,
    dryrun: bool = False,
    verbose: bool = False,
):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Calculating Paths%%%%%%%%%%%%%%%%%%%%%%%%
    voc_root = Path(voc_root)
    image_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"

    anno_dir = voc_root / "ImageSets/Segmentation"
    train_file = anno_dir / "train.txt"
    val_file = anno_dir / "val.txt"

    output_root = Path(output_root)
    if not dryrun:
        # Fail early if we do not have write permission in the directory
        output_root.mkdir(parents=True, exist_ok=True)

    train_image_ids = train_file.read_text().splitlines()
    print(f"Found {len(train_image_ids)} training images.")

    val_image_ids = val_file.read_text().splitlines()
    print(f"Found {len(val_image_ids)} validation images.")

    # %%%%%%%%%%%%%%%%%%%%%Binarize and Save Masks%%%%%%%%%%%%%%%%%%%%%%%%%

    if skip_mask_save:
        print("Skipping mask saving")
    else:
        mask_output_root = output_root / "annotations"
        binarize_masks(
            mask_dir,
            mask_output_root,
            train_image_ids,
            val_image_ids,
            max_workers,
            dryrun,
            verbose,
        )

    if skip_image_save:
        print("Skipping image saving")
        return

    # %%%%%%%%%%%%%%%Save Images to Proper Directories%%%%%%%%%%%%%%%%%%%%%%
    image_output_root = output_root / "images"

    train_image_output_dir = image_output_root / "training"
    val_image_output_dir = image_output_root / "validation"
    if not dryrun:
        train_image_output_dir.mkdir(parents=True, exist_ok=True)
        val_image_output_dir.mkdir(exist_ok=True)

    for image_id in tqdm(train_image_ids, desc="Copying training images"):
        basename = f"{image_id}.jpg"
        in_image_file = image_dir / basename
        out_image_file = train_image_output_dir / basename
        if not dryrun and not out_image_file.is_file():
            shutil.copy(in_image_file, out_image_file)

    for image_id in tqdm(val_image_ids, desc="Copying validation images"):
        basename = f"{image_id}.jpg"
        in_image_file = image_dir / basename
        out_image_file = val_image_output_dir / basename
        if not dryrun and not out_image_file.is_file():
            shutil.copy(in_image_file, out_image_file)


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    VOCDEVKIT_ROOT = Path("/mnt/Enterprise/PUBLIC_DATASETS/VOCdevkit")

    parser = ArgumentParser(
        description="Binarize PascalVOC2012 masks and save images to proper directories."
    )
    parser.add_argument(
        "--voc-root",
        type=str,
        default=VOCDEVKIT_ROOT / "VOC2012",
        help="Root directory of PascalVOC2012 dataset",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=VOCDEVKIT_ROOT / "VOC2012_binarized",
        help="Root directory to save binarized masks and images",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of workers for multiprocessing.",
    )
    parser.add_argument(
        "--skip-mask-save",
        action=BooleanOptionalAction,
        required=True,
        help="Skip mask saving.",
    )
    parser.add_argument(
        "--skip-image-save",
        action=BooleanOptionalAction,
        required=True,
        help="Skip image saving.",
    )
    parser.add_argument(
        "--dryrun",
        action=BooleanOptionalAction,
        required=True,
        help="Dry run mode.  Will not save files.",
    )
    parser.add_argument(
        "--verbose", action=BooleanOptionalAction, required=True, help="Verbose mode."
    )

    args = parser.parse_args()

    main(**vars(args))
