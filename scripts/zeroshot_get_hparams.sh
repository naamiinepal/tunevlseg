#!/bin/bash

source .venv/bin/activate

N=3
for experiment in clip biomedclip; do
	for alpha in $(seq 0 0.05 1); do
		for beta in $(seq 0 0.05 1); do

			# Echo in purple color
			echo -e "\e[1;35mRunning hparams search for $experiment with alpha = $alpha, beta = $beta\e[0m"

			nohup python src/eval.py experiment=zsseg_${experiment}.yaml \
				ckpt_path=null disable_ckpt=true \
				extras.print_config=false \
				model.net.alpha=${alpha} model.net.beta=${beta} \
				model.net.force_no_load_models=true \
				model.net.read_cache=true model.net.write_cache=false \
				data.num_workers=2 \
				trainer.accelerator="cpu" >/dev/null &

			# Sleep some time so that logs do not collide
			sleep 1.5

			# Check if there are 3 or more background jobs running or if accelerator is 'auto'
			if [ $(jobs -r | wc -l) -ge $N ]; then
				wait $(jobs -r -p | head -1) # Wait for the oldest background job
			fi

		done
	done
done

# Wait for any remaining background jobs
wait
