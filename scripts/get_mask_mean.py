import concurrent.futures
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

mask_dir = Path("/mnt/SSD1/rabin/datasets/phrasecut/masks/")


def get_img_mean_from_task(task) -> None:
    # task_id-phrase.png
    task_id = task["task_id"]
    phrase = task["phrase"].replace("/", "\\")
    mask_name = f"{task_id}-{phrase}.png"

    mask_path = mask_dir / mask_name

    img = cv2.imread(str(mask_path))

    return img.mean()


task_path = Path("/mnt/SSD1/rabin/datasets/phrasecut/filtered_tasks/refer_train.json")
with task_path.open() as f:
    tasks = json.load(f)


img_means = []


postfix_update_iter = 5000

print("Spawning threads...")

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    future_to_task = {
        executor.submit(get_img_mean_from_task, task): task for task in tasks
    }
    with tqdm(
        concurrent.futures.as_completed(future_to_task),
        total=len(tasks),
    ) as pbar:
        for i, future in enumerate(pbar):
            task = future_to_task[future]
            try:
                curr_mean = future.result()
            except Exception as exc:
                print(f"Task: {task} generated an exception: {exc}")
            else:
                img_means.append(curr_mean)

                if i % postfix_update_iter == 0:
                    pbar.set_postfix_str(f"Mean: {curr_mean:6.2f}")


overall_mean = np.mean(img_means)

print(f"Overall mean: {overall_mean}")
