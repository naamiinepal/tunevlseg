#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[0]'
batch_size=32
use_new_last_layer=false
precision=16-mixed

ds_name=VOC2012_binarized
.venv/bin/python src/train.py -m hparams_search=cocoop_optuna experiment=coop/cris model=cocoop/cris data=image_dir_text_mask_jpg \
	trainer.devices=$devices trainer.log_every_n_steps=17 \
	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 model.net.use_new_last_layer=$use_new_last_layer \
	+trainer.benchmark=true trainer.precision=$precision
