from __future__ import annotations

import concurrent.futures
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, TypeVar, get_args

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from cv2.typing import MatLike

    StrPath = str | Path

    PolygonType = list[list[float]]
    TaskType = dict[str, int | str | PolygonType]
    RefT = TypeVar("RefT", bound=Mapping[str, Any])

# Used by the argument parser
SplitType = Literal["train", "val", "test", "testA", "testB"]


def read_image(image_path: str):
    """Read image from a path.

    Args:
    ----
        image_path: The image path to read.

    Raises:
    ------
        ValueError: If the image cannot be read.

    Returns:
    -------
        The image in `MatLike` format.
    """
    # Need to load image from cv2 to get the shape.
    # Maybe change to PIL for efficiency
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        msg = f"Can not read image from: {image_path}"
        raise ValueError(msg)

    return img


def get_mask_from_polygons(
    mask_shape: Sequence[SupportsIndex] | SupportsIndex,
    polygons: Iterable[Sequence],
):
    """Convert polygons to an image of type `np.ndarray`.

    Args:
    ----
        mask_shape: The shape of the mask to create.
        polygons: The polygons to draw on the mask.

    Returns:
    -------
        The mask in `np.ndarray` format.
    """
    # Create an empty mask for the polygon
    # Its type can be either float32 or uint8
    mask = np.zeros(mask_shape, np.uint8)

    pts = [np.around(poly).reshape(-1, 2).astype(np.int32) for poly in polygons]
    cv2.fillPoly(mask, pts, 255)  # type:ignore typing needs last argument to be a collections

    return mask


def get_output_name(image_id: object, ann_id: object, sent_id: object) -> str:
    """Get the output path for the mask.

    Args:
    ----
        image_id: The image id to use.
        ann_id: The annotation id to use.
        sent_id: The sentence id to use.

    Returns:
    -------
        The output path for the mask.
    """
    return f"{image_id}-{ann_id}-{sent_id}.png"


def write_mask(
    output_path: Path,
    mask: MatLike,
    task: object,
    skip_mask_if_exists: bool,
    verbose: bool,
) -> None:
    """Write the provided mask to disk.

    Args:
    ----
        output_path: The path to save the mask.
        mask: The mask to save.
        task: The task that the mask is for. It is needed to make exception thrown readable.
        skip_mask_if_exists: Whether to skip the mask if it already exists.
        verbose: Whether to print the saving of files, even if everything is working well.

    Raises:
    ------
        RuntimeError: If the mask cannot be saved to disk.
    """
    if skip_mask_if_exists and output_path.exists():
        if verbose:
            print("Skipping mask creation because it already exists:", output_path)
        return

    try:
        result = cv2.imwrite(str(output_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    except cv2.error as e:
        msg = f"Got an exception while saving the file: {e} for task: {task}"
        raise RuntimeError(msg) from e

    if not result:
        msg = f"Can not save mask to path {output_path}."
        raise RuntimeError(msg)


def process_task(
    task: Mapping[str, Any],
    image_root: Path,
    mask_output_dir: Path,
    skip_mask_if_exists: bool,
    verbose: bool,
):
    """Process a single task. Saves the mask using the provided path.

    Args:
    ----
        task: The task containing `image_id`, `Polygons` and `phrase`.
        image_root: The root directory of the images.
        mask_output_dir: The output directory for the masks.
        skip_mask_if_exists: Whether to skip the mask if it already exists.
        verbose: Whether to print the saving of files, even if everything is working well.

    Raises:
    ------
        ValueError: If the image cannot be read.
        RuntimeError: If the mask cannot be saved to disk.

    Returns:
    -------
        The output path for the mask.
    """
    image_name: str = task["image_name"]

    img = read_image(str(image_root / image_name))

    mask_shape = img.shape[:-1]

    polygons: PolygonType = task["Polygons"]

    mask = get_mask_from_polygons(mask_shape, polygons)

    output_mask_name = get_output_name(
        task["image_id"],
        task["ann_id"],
        task["sent_id"],
    )

    output_path = mask_output_dir / output_mask_name

    write_mask(
        output_path,
        mask,
        task,
        skip_mask_if_exists=skip_mask_if_exists,
        verbose=verbose,
    )

    return output_path


def save_filtered_tasks(tasks: Iterable[Mapping], task_output_path: Path) -> bool:
    """Save the filtered tasks to disk.

    Args:
    ----
        tasks: The original tasks to filter.
        task_output_path: The output path for the filtered tasks.

    Returns:
    -------
        Status of whether the tasks were saved successfully.
    """
    if task_output_path.exists():
        ip = input(
            f"\n{task_output_path} already exists: Do you want to overwrite it? (y/N): ",
        )

        if ip.lower() != "y":
            return False

    filtered_tasks = [
        {k: v for k, v in task.items() if k != "Polygons"} for task in tasks
    ]

    with task_output_path.open("w") as of:
        json.dump(filtered_tasks, of)

    return True


def filter_refs(refs: Iterable[RefT], split: object) -> tuple[RefT, ...]:
    """Filter the refs based on the provided split.

    Args:
    ----
        refs: The refs to filter.
        split: The split to filter on.

    Returns:
    -------
        Filtered refs.
    """
    if split == "test":
        # Return both testA and testB
        return tuple(ref for ref in refs if ref["split"].startswith("test"))
    if split in ("train", "val", "testA", "testB"):
        return tuple(ref for ref in refs if ref["split"] == split)
    return tuple(refs)


def get_image_id2image_name(image_metadata: Iterable[Mapping[str, Any]]):
    """Obtain the image_id to image_name mapping.

    Args:
    ----
        image_metadata: The metadata for the images.

    Returns:
    -------
        The mapping from image_id to image_name.
    """
    image_id2image_name: dict[int, str] = {}
    for img_meta in image_metadata:
        image_id: int = img_meta["id"]
        image_name: str = img_meta["file_name"]
        image_id2image_name[image_id] = image_name
    return image_id2image_name


def get_ann_id2polygons(annotations: Iterable[Mapping[str, Any]]):
    """Get a mapping from ann_id to the polygons in the annotation.

    Args:
    ----
        annotations: The annotations to process.

    Returns:
    -------
        Mapping from ann_id to the polygons.
    """
    ann_id2polygons: dict[int, PolygonType] = {}
    for ann in annotations:
        ann_id: int = ann["id"]
        polygons: PolygonType = ann["segmentation"]
        ann_id2polygons[ann_id] = polygons
    return ann_id2polygons


def get_refs(ref_file_path: StrPath, split: object):
    """Get filtered refs from the pickle file.

    Args:
    ----
        ref_file_path: The full path to the pickle file.
        split: The split to filter on.

    Returns:
    -------
        The refs filtered on the split.
    """
    with open(ref_file_path, "rb") as f:
        refs: list[dict[str, Any]] = pickle.load(f)

    if not refs:
        print("No refs found in:", ref_file_path)
        return None

    split_filtered_refs = filter_refs(refs, split)

    if not split_filtered_refs:
        print("No tasks found for the provided splits.")
        return None
    return split_filtered_refs


def get_instances(instances_json_path: StrPath):
    """Read the instances from the json file.

    Args:
    ----
        instances_json_path: The full path to the json file.

    Returns:
    -------
        The instances extracted from the json file.
    """
    with open(instances_json_path) as f:
        instances: dict[str, Any] = json.load(f)

    if not instances_json_path:
        print("No instances provided in:", instances_json_path)
        return None
    return instances


def get_tasks(instances_json_path: StrPath, ref_file_path: StrPath, split: object):
    """Obtain the tasks from the instances, refs, and split.

    Args:
    ----
        instances_json_path: The path to the instances json.
        ref_file_path: The path to the ref file.
        split: The split to filter on.

    Returns:
    -------
        The tasks containing image_id, ann_id, sent_id, phrase, and polygons.
        Returns None if there are no tasks.
    """
    refs = get_refs(ref_file_path, split)

    if refs is None:
        return None

    instances = get_instances(instances_json_path)

    if instances is None:
        return None

    image_id2image_name = get_image_id2image_name(instances["images"])

    ann_id2polygons = get_ann_id2polygons(instances["annotations"])

    tasks: list[TaskType] = []
    for ref in refs:
        ann_id: int = ref["ann_id"]
        image_id: int = ref["image_id"]
        image_name: str = image_id2image_name[image_id]
        polygons: PolygonType = ann_id2polygons[ann_id]
        for sent_metadata in ref["sentences"]:
            sent: str = sent_metadata["sent"]
            sent_id: int = sent_metadata["sent_id"]

            tasks.append(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "ann_id": ann_id,
                    "sent_id": sent_id,
                    "phrase": sent,
                    "Polygons": polygons,
                },
            )
    return tuple(tasks)


def get_dataset(ref_file_path: Path):
    """Get dataset from the stem of the path to the ref file.
    It should be in the format: `refs(dataset).p`.

    Args:
    ----
        ref_file_path: The path to the ref file.

    Raises:
    ------
        RuntimeError: The ref file stem is not in the expected format.

    Returns:
    -------
        The dataset extracted from the ref file stem.
    """
    ref_file_stem = ref_file_path.stem[:-1]
    splitted = ref_file_stem.split("(", 1)
    if len(splitted) != 2:
        msg = (
            "The Ref File Path is not valid."
            "It should be in the format: 'refs(dataset).p'",
        )
        raise RuntimeError(msg)
    return splitted[1]


def get_unique_masks(
    futures_to_task: Mapping[concurrent.futures.Future[Path], object],
    verbose: bool,
):
    """Get unique mask paths from the futures.

    Args:
    ----
        futures_to_task: The mapping of futures to metadata.
        verbose: Whether to print the saving of files, even if everything is working well.

    Returns:
    -------
        The unique mask paths.
    """
    unique_mask_paths: set[str] = set()
    for future in concurrent.futures.as_completed(futures_to_task):
        try:
            mask_path = future.result()
        except Exception as e:
            image_id = futures_to_task[future]
            print(f"Got an exception: {e} for image_id: {image_id}")
        else:
            if verbose:
                print(f"Saved mask to path {mask_path}.")

            unique_mask_paths.add(mask_path.name)
    return unique_mask_paths


def get_created_dir(output_dir: StrPath):
    """Creates a directory and its parents if it does not exist.

    Args:
    ----
        output_dir: The path to the output directory to creae.

    Returns:
    -------
        The created directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_preamble(
    image_root: object,
    instances_json_path: object,
    mask_output_dir: object,
    ref_file_path: object,
    split: object,
    task_output_dir: object,
) -> None:
    """Prints the preamble for the current run.

    Args:
    ----
        image_root: The root of the image files.
        instances_json_path: The path to the instances json.
        mask_output_dir: The directory to save the masks.
        ref_file_path: The file path to the ref file.
        split: The split to filter on.
        task_output_dir: The output directory for the tasks.
    """
    print("\nRef File Path:", ref_file_path)
    print("Split:", split)
    print("Instances JSON Path:", instances_json_path)
    print("Image root", image_root)
    print("Mask Output Directory", mask_output_dir)
    print("Task Output Directory", task_output_dir)


def main(
    ref_file_path: StrPath,
    split: SplitType | None,
    instances_json_path: StrPath,
    image_root: StrPath,
    mask_output_dir: StrPath,
    task_output_dir: StrPath,
    max_workers: int | None,
    skip_mask_if_exists: bool,
    verbose: bool,
) -> None:
    image_root = Path(image_root)

    if not image_root.is_dir():
        msg = f"{image_root} is not a directory"
        raise ValueError(msg)

    mask_output_dir = get_created_dir(mask_output_dir)

    ref_file_path = Path(ref_file_path)

    dataset = get_dataset(ref_file_path)
    task_output_dir = get_created_dir(Path(task_output_dir, dataset))

    print_preamble(
        image_root,
        instances_json_path,
        mask_output_dir,
        ref_file_path,
        split,
        task_output_dir,
    )

    tasks = get_tasks(instances_json_path, ref_file_path, split)

    if tasks is None:
        return

    filtered_task_save_path = task_output_dir / f"{split}.json"

    save_filtered_tasks(tasks, filtered_task_save_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        futures_to_task = {
            executor.submit(
                process_task,
                task,
                image_root,
                mask_output_dir,
                skip_mask_if_exists,
                verbose,
            ): task["image_id"]
            for task in tasks
        }

        unique_mask_paths = get_unique_masks(futures_to_task, verbose)

    num_tasks = len(tasks)
    num_unique_mask_paths = len(unique_mask_paths)

    print("Original number of tasks:", num_tasks)
    print("Unique masks saved:", num_unique_mask_paths)

    if num_tasks != num_unique_mask_paths:
        msg = (
            f"The number of masks saved: {num_unique_mask_paths} and the number of "
            f"tasks: {num_tasks} are not equal for ref file: {ref_file_path} "
            f"and instance json: {instances_json_path}"
        )
        raise ValueError(msg)


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(
        description="Load the task JSON, save the masks generated from the polygons provided in the tasks, and save the filtered JSON in the provided directory.",
    )

    parser.add_argument(
        "--ref-file-path",
        required=True,
        help="The path to the pickle file containing the tasks.",
    )

    parser.add_argument(
        "--split",
        default=None,
        choices=get_args(SplitType),
        help="The split of the dataset to filter.",
    )

    parser.add_argument(
        "--instances-json-path",
        required=True,
        help="The path to the JSON file containing the image and annotation info.",
    )

    parser.add_argument(
        "--image-root",
        required=True,
        help="The directory containing the images in a flat structure.",
    )

    parser.add_argument(
        "--mask-output-dir",
        required=True,
        help="The directory to output the masks.",
    )

    parser.add_argument(
        "--task-output-dir",
        required=True,
        help="The directory to output the filtered tasks.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="The maximum number of workers to use for multithreading.",
    )

    parser.add_argument(
        "--skip-mask-if-exists",
        action=BooleanOptionalAction,
        default=True,
        help="Whether to skip saving the mask if it already exists.",
    )

    parser.add_argument(
        "--verbose",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to print the saving of files, even if everything is working well.",
    )

    args = parser.parse_args()

    main(**vars(args))
