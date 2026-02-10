#!/bin/bash

python cs224r/scripts/run_hw1.py \
  --expert_policy_file cs224r/policies/experts/HalfCheetah.pkl \
  --env_name HalfCheetah-v4 \
  --exp_name bc_cheetah \
  --n_iter 10 \
  --expert_data cs224r/expert_data/expert_data_HalfCheetah-v4.pkl \
  --batch_size 5000 \
  --eval_batch_size 10000 \
  --scalar_log_freq 1 \
  --video_log_freq -1