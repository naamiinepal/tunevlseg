#! /bin/bash

for experiment in clip biomedclip; do
	for alpha in $(seq 0 0.05 1); do
		for beta in $(seq 0 0.05 1); do
			.venv/bin/python src/eval.py experiment=zsseg_${experiment}.yaml \
				ckpt_path=null disable_ckpt=true trainer.devices=auto \
				model.net.alpha=${alpha} model.net.beta=${beta}
		done
	done
done
