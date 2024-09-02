#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[0]'
batch_size=32
# accumulate_grad_batches=2
precision=16-mixed

ds_name="busi"

# .venv/bin/python src/train.py -m hparams_search=maple_optuna experiment=coop/clipseg model=maple_clipseg \
# 	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=4 \
# 	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
#
# .venv/bin/python src/train.py -m hparams_search=shared_separate_optuna experiment=coop/clipseg model=shared_separate_clipseg \
# 	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=4 \
# 	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
#
# .venv/bin/python src/train.py -m hparams_search=shared_attn_optuna experiment=coop/clipseg model=shared_attn_clipseg \
# 	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=4 \
# 	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
#
# .venv/bin/python src/train.py -m hparams_search=vpt_optuna experiment=coop/clipseg.yaml model=vpt_clipseg \
# 	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=4 \
# 	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
#
# .venv/bin/python src/train.py -m hparams_search=cocoop_optuna experiment=coop/cris model=cocoop/cris \
# 	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=4 \
# 	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches
#
# .venv/bin/python src/train.py -m hparams_search=coop_optuna experiment=coop/cris model=coop/cris \
# 	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=4 \
# 	+trainer.benchmark=true trainer.precision=$precision +trainer.accumulate_grad_batches=$accumulate_grad_batches

.venv/bin/python src/train.py -m hparams_search=cocoop_optuna experiment=coop/clipseg model=cocoop/clipseg \
	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
	+trainer.benchmark=true trainer.precision=$precision

.venv/bin/python src/train.py -m hparams_search=coop_optuna experiment=coop/clipseg model=coop/clipseg \
	prompt_index=1 trainer.devices=$devices trainer.log_every_n_steps=3 \
	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
	+trainer.benchmark=true trainer.precision=$precision
