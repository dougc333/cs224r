#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:3000}"
ALGO="${ALGO:-PPO}"
TIMESTEPS="${TIMESTEPS:-1000000}"
N_ENVS="${N_ENVS:-8}"

ENVS=(Ant-v5 Humanoid-v5 HalfCheetah-v5 Walker2d-v5 Hopper-v5 Swimmer-v5 Reacher-v5 Pusher-v5)
for ENV_ID in "${ENVS[@]}"; do
  curl -sS -X POST "${BASE_URL}/api/jobs/create" \
    -H 'Content-Type: application/json' \
    -d "{\"envId\":\"${ENV_ID}\",\"algo\":\"${ALGO}\",\"totalTimesteps\":${TIMESTEPS},\"nEnvs\":${N_ENVS}}" | jq .
done
