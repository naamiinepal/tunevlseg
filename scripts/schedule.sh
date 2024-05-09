#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

std=0.02
devices='[0]'
use_proj_norm=true
prompt_depth=11
batch_size=32
lr=5.0e-4
weight_decay=1.0e-3
intermediate_dim=64
precision=16-mixed
# context_initializer=null
num_context=4
use_new_last_layer=true

# bkai_polyp clinicdb_polyp kvasir_polyp busi chexlocalize dfu isic
for ds_name in bkai_polyp clinicdb_polyp kvasir_polyp busi chexlocalize dfu isic; do
	# Enable auto-tuner for cudnn may increase performance in the cost of memory
	.venv/bin/python src/train.py experiment=coop/clipseg.yaml model=maple_clipseg.yaml \
		prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
		ds_name=$ds_name model.weight_decay=$weight_decay +model.net.no_freeze_last_layer=true \
		model.net.context_learner.context_initializer="a photo of a" model.net.context_learner.prompt_depth=$prompt_depth \
		model.net.context_learner.use_unified_projection=false model.net.context_learner.intermediate_dim=$intermediate_dim \
		model.net.context_learner.vector_std=$std model.net.context_learner.use_proj_norm=$use_proj_norm \
		model.net.context_learner.num_context=$num_context +model.net.use_new_last_layer=$use_new_last_layer \
		data.batch_size=$batch_size data.num_workers=8 model.optimizer.lr=$lr \
		+trainer.benchmark=true trainer.precision=$precision
done

ds_name=camus
.venv/bin/python src/train.py experiment=coop/clipseg.yaml model=maple_clipseg.yaml data=image_text_mask_camus \
	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
	ds_name=$ds_name model.weight_decay=$weight_decay +model.net.no_freeze_last_layer=true \
	model.net.context_learner.context_initializer="a photo of a" model.net.context_learner.prompt_depth=$prompt_depth \
	model.net.context_learner.use_unified_projection=false model.net.context_learner.intermediate_dim=$intermediate_dim \
	model.net.context_learner.vector_std=$std model.net.context_learner.use_proj_norm=$use_proj_norm \
	model.net.context_learner.num_context=$num_context +model.net.use_new_last_layer=$use_new_last_layer \
	data.batch_size=$batch_size data.num_workers=8 model.optimizer.lr=$lr \
	+trainer.benchmark=true trainer.precision=$precision
