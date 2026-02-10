#!/bin/bash

python cs224r/scripts/run_hw1.py \
	--expert_policy_file cs224r/policies/experts/Ant.pkl \
	--env_name Ant-v4 \
	--exp_name bc_ant \
	--n_iter 20 \
	--ep_len 1000 \
	--learning_rate 5e-3 \
	--batch_size 5000 \
	--eval_batch_size 10000 \
	--scalar_log_freq 1 \
	--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq 1

