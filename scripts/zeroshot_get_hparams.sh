#! /bin/bash

for experiment in clip biomedclip; do
	accelerator='auto'
	force_no_load_models=false
	for alpha in $(seq 0 0.05 1); do
		for beta in $(seq 0 0.05 1); do
			# Echo in purple color
			echo -e "\e[1;35mRunning hparams search for $experiment with alpha = $alpha, beta = $beta\e[0m"

			.venv/bin/python src/eval.py experiment=zsseg_${experiment}.yaml \
				ckpt_path=null disable_ckpt=true trainer.devices=auto \
				model.net.alpha=${alpha} model.net.beta=${beta} \
				model.force_no_load_models=${force_no_load_models} \
				trainer.accelerator=${accelerator}

			accelerator='cpu'
			force_no_load_models=true
		done
	done
done
