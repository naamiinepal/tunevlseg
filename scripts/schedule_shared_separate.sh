#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[0]'
batch_size=32
precision=16-mixed

# bkai_polyp clinicdb_polyp kvasir_polyp busi chexlocalize dfu isic
for ds_name in bkai_polyp clinicdb_polyp kvasir_polyp busi chexlocalize dfu isic; do
	# Enable auto-tuner for cudnn may increase performance in the cost of memory
	.venv/bin/python src/train.py -m hparams_search=shared_separate_optuna experiment=coop/clipseg model=shared_separate_clipseg \
		prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
		ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
		+trainer.benchmark=true trainer.precision=$precision
done

ds_name=camus
.venv/bin/python src/train.py -m hparams_search=shared_separate_optuna experiment=coop/clipseg model=shared_separate_clipseg data=image_text_mask_camus \
	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
	+trainer.benchmark=true trainer.precision=$precision
