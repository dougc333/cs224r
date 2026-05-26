# MuJoCo RL GKE Dashboard

This scaffold deploys:

- A Python RL trainer image using Gymnasium MuJoCo + Stable-Baselines3.
- Kubernetes Jobs for Ant, Humanoid, HalfCheetah, Walker2d, Hopper, Swimmer, Reacher, and Pusher.
- Prometheus metrics from each training pod.
- A Next.js dashboard that launches jobs, shows status, logs, and reward/loss/episode-length graphs.

## Metrics: Prometheus or React?

Use Prometheus for time-series metrics:

- `rl_episode_return_mean`
- `rl_episode_return_min`
- `rl_episode_length_mean`
- `rl_episode_length_min`
- `rl_training_steps`
- `rl_train_loss{name=...}`
- container CPU/memory and Kubernetes pod/job metrics from kube-prometheus-stack

React should only query and visualize. Later, add Postgres for durable experiment metadata and GCS/S3 for artifacts.

## Quick deploy

```bash
cd infra/scripts
export PROJECT_ID=your-gcp-project
export REGION=us-central1
./deploy_gke.sh
```

Then wait for the external IP:

```bash
kubectl -n rl get svc rl-dashboard -w
```

## Local dashboard dev

```bash
cd dashboard
npm install
npm run dev
```

For local Kubernetes access, your `kubectl` context must point to the cluster.

## Launch all envs from dashboard

Open the dashboard and press **Launch all 8 envs**.

Or call the API:

```bash
BASE_URL=http://DASHBOARD_IP ./infra/scripts/launch_all_envs.sh
```

## Environments

- `Ant-v5`
- `Humanoid-v5`
- `HalfCheetah-v5`
- `Walker2d-v5`
- `Hopper-v5`
- `Swimmer-v5`
- `Reacher-v5`
- `Pusher-v5`

## Notes

Normal Gymnasium MuJoCo runs physics on CPU. For GPU physics, use MJX/Warp/Brax/Isaac-style environments rather than classic Gymnasium MuJoCo.
