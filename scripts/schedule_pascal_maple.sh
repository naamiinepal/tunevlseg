#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[1]'
batch_size=16
accumulate_grad_batches=2
precision=16-mixed

ds_name=VOC2012_binarized
.venv/bin/python src/train.py -m hparams_search=maple_optuna experiment=coop/clipseg model=maple_clipseg data=image_dir_text_mask_jpg \
	trainer.devices=$devices trainer.log_every_n_steps=17 \
	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
	model.net.context_learner.context_initializer=null \
	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
