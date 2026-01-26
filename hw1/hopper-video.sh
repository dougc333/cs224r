#!/bin/bash

python cs224r/scripts/run_hw1.py \
	--expert_policy_file cs224r/policies/experts/Hopper.pkl \
	--env_name Hopper-v4 \
	--ep_len 1000 \
	--n_iter 20 \
	--exp_name bc_hopper \ 
	--n_iter 20 \
	--expert_data cs224r/expert_data/expert_data_Hopper-v4.pkl \
	--batch_size 5000 \
	--eval_batch_size 10000 \
	--scalar_log_freq 1 \
	--video_log_freq 1

