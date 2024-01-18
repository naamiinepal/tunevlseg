#! /bin/bash

source .venv/bin/activate

ROOT_DIR=/mnt/SSD1/rabin/datasets/phrasecut

for task_file in $ROOT_DIR/refer*.json; do
	python scripts/reduce_and_save_phrasecut.py \
		--task-json-path $task_file \
		--image-root $ROOT_DIR/images \
		--mask-output-dir $ROOT_DIR/masks \
		--task-output-dir $ROOT_DIR/filtered_tasks
done
