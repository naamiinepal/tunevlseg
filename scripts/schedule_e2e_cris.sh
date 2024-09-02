#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[0]'
batch_size=32
accumulate_grad_batches=4
num_workers=12
precision=16-mixed

ds_name=VOC2012_binarized
.venv/bin/python src/train.py experiment=e2e_cris model=e2e_cris data=image_dir_text_mask_jpg \
  trainer.devices=$devices trainer.log_every_n_steps=4 \
  ds_name=$ds_name data.batch_size=$batch_size data.num_workers=$num_workers \
  +trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches

ds_name=cityscapes_binarized
.venv/bin/python src/train.py experiment=e2e_cris model=e2e_cris data=image_dir_text_mask_png \
  trainer.devices=$devices trainer.log_every_n_steps=120 +trainer.val_check_interval=0.5 \
  ds_name=$ds_name data.batch_size=$batch_size data.num_workers=$num_workers \
  +trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
