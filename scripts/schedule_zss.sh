#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Turn off parallelism for huggingface tokenizers
export TOKENIZERS_PARALLELISM=false

# Use when you're low in memory
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

devices='[0]'
batch_size=32
precision=32

# bkai_polyp clinicdb_polyp kvasir_polyp busi chexlocalize dfu isic
ds_name="cityscapes_binarized"
# .venv/bin/python src/eval.py experiment=coop/clipseg model=clipseg_zss data=image_dir_text_mask_png \
# 	trainer.devices=$devices ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# 	trainer.precision=$precision ckpt_path=null disable_ckpt=true
#
.venv/bin/python src/eval.py experiment=coop/cris model=cris_zss data=image_dir_text_mask_png \
	trainer.devices=$devices ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
	logger='' testing=false predict=true \
	trainer.precision=$precision ckpt_path=null disable_ckpt=true

# ds_name="VOC2012_binarized"
# # .venv/bin/python src/eval.py experiment=coop/clipseg model=clipseg_zss data=image_dir_text_mask_jpg \
# # 	trainer.devices=$devices ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# # 	trainer.precision=$precision ckpt_path=null disable_ckpt=true
# #
# .venv/bin/python src/eval.py experiment=coop/cris model=cris_zss data=image_dir_text_mask_jpg \
# 	trainer.devices=$devices ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# 	trainer.precision=$precision ckpt_path=null disable_ckpt=true
#
# for ds_name in bkai_polyp clinicdb_polyp kvasir_polyp busi chexlocalize dfu isic; do
# 	# .venv/bin/python src/eval.py experiment=coop/clipseg model=clipseg_zss \
# 	# 	prompt_index=1 trainer.devices=$devices \
# 	# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# 	# 	trainer.precision=$precision ckpt_path=null disable_ckpt=true
#
# 	.venv/bin/python src/eval.py experiment=coop/cris model=cris_zss \
# 		prompt_index=1 trainer.devices=$devices \
# 		ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# 		trainer.precision=$precision ckpt_path=null disable_ckpt=true
# done
#
# ds_name=camus
#
# # .venv/bin/python src/eval.py experiment=coop/clipseg model=clipseg_zss data=image_text_mask_camus \
# # 	prompt_index=1 trainer.devices=$devices \
# # 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# # 	trainer.precision=$precision ckpt_path=null disable_ckpt=true
#
# .venv/bin/python src/eval.py experiment=coop/cris model=cris_zss data=image_text_mask_camus \
# 	prompt_index=1 trainer.devices=$devices \
# 	ds_name=$ds_name data.batch_size=$batch_size data.num_workers=8 \
# 	trainer.precision=$precision ckpt_path=null disable_ckpt=true
