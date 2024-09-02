#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

std=0.02
devices='[0]'
prompt_depth=11
batch_size=32
lr=5.0e-4
weight_decay=1.0e-3
precision=16-mixed
# context_initializer=null
num_context=4
use_new_last_layer=true
num_workers=4

ds_name=kvasir_polyp
# for model in coop/clipseg cocoop/clipseg maple_clipseg shared_attn_clipseg shared_separate_clipseg vpt_clipseg; do
# 	.venv/bin/python src/train.py experiment=coop/clipseg.yaml model=$model.yaml \
# 		prompt_index=1 trainer.devices=$devices \
# 		ds_name=$ds_name model.weight_decay=$weight_decay \
# 		model.net.context_learner.prompt_depth=$prompt_depth \
# 		model.net.context_learner.vector_std=$std \
# 		model.net.context_learner.num_context=$num_context model.net.use_new_last_layer=$use_new_last_layer \
# 		data.batch_size=$batch_size data.num_workers=$num_workers model.optimizer.lr=$lr \
# 		trainer.precision=$precision +trainer.fast_dev_run=true
# done

# for model in coop/cris cocoop/cris; do
for model in cocoop/cris; do
	.venv/bin/python src/train.py experiment=coop/cris.yaml model=$model.yaml \
		prompt_index=1 trainer.devices=$devices \
		ds_name=$ds_name model.weight_decay=$weight_decay \
		model.net.context_learner.context_initializer="a photo of a" model.net.context_learner.prompt_depth=$prompt_depth \
		model.net.context_learner.vector_std=$std \
		model.net.context_learner.num_context=$num_context model.net.use_new_last_layer=$use_new_last_layer \
		data.batch_size=$batch_size data.num_workers=$num_workers model.optimizer.lr=$lr \
		trainer.precision=$precision +trainer.fast_dev_run=true +trainer.detect_anomaly=true
done
