"""Script to evaluate the segmentation metrics.
The script takes in the path to the segmentation and ground truth images and
computes the following metrics:
1. Surface Dice
2. Hausdorff Distance
3. IoU
4. Dice.

NOTE: The script assumes that the segmentation and ground truth images
have the same name. The script also assumes that the images are binary
images with pixel values 0 and 255. The script thresholds the images to 0 and 1
and computes the metrics. The script also assumes that the images are of size
[H, W] and not [H, W, C], and are of type uint8.

The script saves the metrics in a csv file.

Usage:
    python eval_metrics.py \
        --seg_path <path to segmentation images> \
        --gt_path <path to ground truth images> \
        --csv_path <path to save csv file>
"""

import concurrent.futures
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou
from tqdm import tqdm


def load_image(image_path: str, flags: int = cv2.IMREAD_GRAYSCALE):
    image = cv2.imread(image_path, flags)

    if image is None:
        msg = f"Image Not found: {image_path}"
        raise ValueError(msg)

    return image


def compute_metrics(gt_img_path: str, pred_img_path: str, threshold: int):
    gt_img = load_image(gt_img_path)
    pred_img = load_image(pred_img_path)

    # make sure the images are of same size
    assert (
        gt_img.shape == pred_img.shape
    ), f"Images {gt_img_path} and {pred_img_path} are of different sizes"

    # threshold the images
    gt_img = gt_img > 127  # type: ignore
    pred_img = pred_img > threshold  # type: ignore

    # change images to batch-first tensor [B,C,H,W]
    gt_img = torch.from_numpy(gt_img)[None, None, ...]
    pred_img = torch.from_numpy(pred_img)[None, None, ...]

    # compute the metrics
    # surface_dice = compute_surface_dice(pred_img, gt_img, class_thresholds=[0.5]) * 100
    # hausdorff_distance = compute_hausdorff_distance(pred_img, gt_img) * 100
    iou = compute_iou(pred_img, gt_img, ignore_empty=False) * 100
    dice = compute_dice(pred_img, gt_img, ignore_empty=False) * 100

    all_ones_pred = torch.ones_like(pred_img)
    all_ones_dice = compute_dice(all_ones_pred, gt_img, ignore_empty=False) * 100
    ones_dice_diff = dice - all_ones_dice

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "ones_dice_diff": ones_dice_diff.item(),
    }


def main(
    seg_path: Path,
    gt_path: Path,
    csv_path: str | Path,
    max_workers: int | None,
    threshold: int,
) -> None:
    np.set_printoptions(precision=5)

    cpu_count = os.cpu_count() or 1

    torch.set_num_threads(cpu_count // (max_workers or cpu_count))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for filename in seg_path.glob("*.png"):
            gt_img_path = str(gt_path / filename.name)
            pred_img_path = str(seg_path / filename.name)

            futures[
                executor.submit(compute_metrics, gt_img_path, pred_img_path, threshold)
            ] = filename

        aggregator = defaultdict(list)

        with tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating metrics",
        ) as pbar:
            for future in pbar:
                filename = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    print(f"{filename} generated an exception: {exc}")
                else:
                    aggregator["filename"].append(filename)
                    for key, value in results.items():
                        aggregator[key].append(value)

                pbar.set_postfix(
                    {
                        "Mean Dice": np.mean(aggregator["dice"]),
                        "Mean IoU": np.mean(aggregator["iou"]),
                    },
                )

    df = pd.DataFrame(aggregator)

    # print mean and std for each metric
    for key in df.columns:
        if key != "filename":
            print_mean_std(df, key)

    # sort the dataframe by filename to make output consistent
    df = df.sort_values(by="filename")

    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved metrics to {csv_path}")


def print_mean_std(df: pd.DataFrame, column_name: str) -> None:
    column = df[column_name]
    print(
        column_name.replace("_", " ").title(),
        "$",
        round(column.mean()),
        r"\smallStd{",
        round(column.std()),
        "}$",
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--seg-path",
        type=Path,
        required=True,
        help="path to segmentation files",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        required=True,
        help="path to ground truth files",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="metrics.csv",
        help="path to save csv file",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="maximum number of workers to use for multiprocessing",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Integer threshold for prediction mask.",
    )

    args = parser.parse_args()

    main(**vars(args))
