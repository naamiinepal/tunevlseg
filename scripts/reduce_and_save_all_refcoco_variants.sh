#! /bin/bash

source .venv/bin/activate

DATASET_ROOT_DIR='/mnt/Enterprise2/PUBLIC_DATASETS'
OUTPUT_ROOT_DIR="$DATASET_ROOT_DIR/refcoco_processed"

for dataset in refcoco refcoco+ refcocog; do
	CURRENT_DATA_DIR="$DATASET_ROOT_DIR/$dataset"
	CURRENT_OUTPUT_DIR="$OUTPUT_ROOT_DIR/$dataset"
	for split in train val test testA testB; do
		for ref_file in $CURRENT_DATA_DIR/refs*.p; do
			python scripts/reduce_and_save_refcoco.py \
				--ref-file-path $ref_file \
				--split $split \
				--instances-json-path $CURRENT_DATA_DIR/instances.json \
				--image-root $DATASET_ROOT_DIR/coco/train2014 \
				--mask-output-dir $CURRENT_OUTPUT_DIR/masks \
				--task-output-dir $CURRENT_OUTPUT_DIR/filtered_tasks
		done
	done
done
