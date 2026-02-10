#!/bin/bash

python cs224r/scripts/run_hw1.py \
	--expert_policy_file cs224r/policies/experts/Walker2d.pkl \
	--env_name Walker2d-v4 --exp_name bc_walker --n_iter 10 \
	--expert_data cs224r/expert_data/expert_data_Walker2d-v4.pkl \
	--video_log_freq -1

