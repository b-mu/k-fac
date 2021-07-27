#!/bin/bash
epochs=90
device=$@

for lr in  1e-1 3e-2 1e-2 3e-3 1e-3 3e-1 
do
	for wd in 3e-3 1e-3 1e-2 3e-2
	do
		for damping in 1e-2 3e-2 1e-1 3e-1
		do
			python main.py --freq 100 --trial true --step_info false --batchnorm false --low_rank true --dataset imagenet --batch_size 128 --device $device --optimizer kngd --network wrn --depth 28 --widen_factor 4 --epoch $epochs --milestone 30,60,90 --learning_rate_decay 0.1 --learning_rate $lr --damping $damping --weight_decay $wd --momentum 0.9
		done
	done
done