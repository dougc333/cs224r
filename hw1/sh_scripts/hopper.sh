#!/bin/bash

python cs224r/scripts/run_hw1.py \
	--expert_policy_file cs224r/policies/experts/Hopper.pkl \
	--env_name Hopper-v4 --exp_name bc_hopper --n_iter 10 \
	--expert_data cs224r/expert_data/expert_data_Hopper-v4.pkl \
	--video_log_freq -1

