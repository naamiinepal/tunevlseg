#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[0]'
batch_size=16
accumulate_grad_batches=2
precision=16-mixed

ds_name=VOC2012_binarized
.venv/bin/python src/train.py -m hparams_search=shared_separate_optuna experiment=coop/clipseg model=shared_separate_clipseg data=image_dir_text_mask_jpg \
	trainer.devices=$devices trainer.log_every_n_steps=17 \
	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 +trainer.accumulate_grad_batches=$accumulate_grad_batches \
	+trainer.benchmark=true trainer.precision=$precision
