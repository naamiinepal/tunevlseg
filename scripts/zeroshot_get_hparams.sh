#!/bin/bash

source .venv/bin/activate

N=3
for experiment in clip biomedclip; do
	accelerator='auto'
	devices='[0]'
	force_no_load_models=false

	for alpha in $(seq 0 0.05 1); do
		for beta in $(seq 0 0.05 1); do

			# Echo in purple color
			echo -e "\e[1;35mRunning hparams search for $experiment with alpha = $alpha, beta = $beta\e[0m"

			nohup python src/eval.py experiment=zsseg_${experiment}.yaml \
				ckpt_path=null disable_ckpt=true \
				extras.print_config=false \
				model.net.alpha=${alpha} model.net.beta=${beta} \
				model.net.force_no_load_models=${force_no_load_models} \
				model.net.threshold_solo_mask=true \
				trainer.accelerator=${accelerator} trainer.devices=${devices} >/dev/null &

			# Check if there are 3 or more background jobs running or if accelerator is 'auto'
			if [ $(jobs -r | wc -l) -ge $N ] || [ $accelerator == 'auto' ]; then
				wait $(jobs -r -p | head -1) # Wait for the oldest background job
			fi

			accelerator='cpu'
			devices='auto'
			force_no_load_models=true
		done
	done
done

# Wait for any remaining background jobs
wait
