from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    StrPath = str | Path

    PolygonType = Iterable[Iterable[Iterable[tuple[int, int]]]]
    TaskType = Mapping[str, str | int | PolygonType]


def process_task(task: TaskType, image_root: Path, mask_output_dir: Path):
    image_id: int = task["image_id"]  # type: ignore

    str_path = str(image_root / f"{image_id}.jpg")

    # Need to load image from cv2 to get the shape.
    # Maybe change to PIL for efficiency
    img = cv2.imread(str_path, cv2.IMREAD_COLOR)

    if img is None:
        msg = f"Can not read image from: {str_path}"
        raise ValueError(msg)

    mask_shape = img.shape[:-1]

    # Create an empty mask for the polygon
    # Its type can be either float32 or uint8
    mask = np.zeros(mask_shape, np.uint8)

    polygons: PolygonType = task["Polygons"]  # type: ignore

    # Loop to add multiple polygons to the mask
    for poly in polygons:
        pts = [np.around(p).astype(np.int32) for p in poly]  # type: ignore
        cv2.fillPoly(mask, pts, 255)  # type: ignore

    phrase: str = task["phrase"]  # type: ignore

    # Use phrase to save the masks to make it readable from file browser
    safe_phrase = phrase.replace("\x00", "").replace("/", "\\")

    mask_name = f"{task['task_id']}-{safe_phrase}.png"

    output_path = mask_output_dir / mask_name

    try:
        result = cv2.imwrite(str(output_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    except cv2.error as e:
        msg = f"Got an exception while saving the file: {e} for task: {task}"
        raise RuntimeError(msg) from e

    if not result:
        msg = f"Can not save mask to path {output_path}."
        raise RuntimeError(msg)

    return output_path


def save_filtered_tasks(tasks: list[TaskType], task_output_path: Path) -> bool:
    # The first split of task_id using __ is the image_id
    # Remove the null characters
    filtered_tasks: dict[str, str] = [
        {k: task[k].replace("\x00", "") for k in ("task_id", "phrase")}  # type: ignore
        for task in tasks
    ]

    if task_output_path.exists():
        ip = input(
            f"\n{task_output_path} already exists: Do you want to overwrite it? (y/N): ",
        )

        if ip.lower() != "y":
            return False

    with task_output_path.open("w") as of:
        json.dump(filtered_tasks, of)

    return True


def main(
    task_json_path: StrPath,
    image_root: StrPath,
    mask_output_dir: StrPath,
    task_output_dir: StrPath,
    max_workers: int | None,
    verbose: bool,
) -> None:
    image_root = Path(image_root)

    if not image_root.is_dir():
        msg = f"{image_root} is not a directory"
        raise ValueError(msg)

    mask_output_dir = Path(mask_output_dir)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    task_output_dir = Path(task_output_dir)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nTask JSON Path:", task_json_path)
    print("Image root", image_root)
    print("Mask Output Directory", mask_output_dir)
    print("Task Output Directory", task_output_dir)

    task_json_path = Path(task_json_path)

    # Error is automatically thrown here
    with task_json_path.open() as f:
        tasks: list[TaskType] = json.load(f)

    if not tasks:
        print("No task provided in:", task_json_path)
        return

    save_filtered_tasks(tasks, task_output_dir / task_json_path.name)

    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        futures_to_task = {
            executor.submit(process_task, task, image_root, mask_output_dir): task[
                "image_id"
            ]
            for task in tasks
        }

        unique_mask_paths: set[str] = set()
        for future in concurrent.futures.as_completed(futures_to_task):
            try:
                mask_path = future.result()
            except Exception as e:  # noqa: PERF203, BLE001
                image_id = futures_to_task[future]
                print(f"Got an exception: {e} for image_id: {image_id}")
            else:
                if verbose:
                    print(f"Saved mask to path {mask_path}.")

                unique_mask_paths.add(mask_path.name)

    num_tasks = len(tasks)
    num_unique_mask_paths = len(unique_mask_paths)

    print("Original number of tasks:", num_tasks)
    print("Unique masks saved:", num_unique_mask_paths)

    if num_tasks != num_unique_mask_paths:
        msg = f"The number of masks saved: {num_unique_mask_paths} and the number of tasks: {num_tasks} are not equal for {task_json_path}."
        raise ValueError(msg)


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(
        description="Load the task JSON, save the masks generated from the polygons provided in the tasks, and save the filtered JSON in the provided directory.",
    )

    parser.add_argument(
        "--task-json-path",
        required=True,
        help="The path to the JSON containing the tasks to load.",
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
        "--verbose",
        action=BooleanOptionalAction,
        default=False,
        help="Whether to print the saving of files, even if everything is working well.",
    )

    args = parser.parse_args()

    main(**vars(args))
