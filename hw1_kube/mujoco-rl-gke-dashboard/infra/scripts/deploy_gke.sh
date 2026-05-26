#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-mujoco-rl-cluster}"
REPO="${REPO:-rl-images}"
TRAINER_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/mujoco-trainer:latest"
DASHBOARD_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/rl-dashboard:latest"

if [[ "$PROJECT_ID" == "your-gcp-project" ]]; then
  echo "Set PROJECT_ID first: export PROJECT_ID=..."
  exit 1
fi

gcloud config set project "$PROJECT_ID"
gcloud services enable container.googleapis.com artifactregistry.googleapis.com compute.googleapis.com

gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="RL images" || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q

# Standard cluster is easier for RL jobs than Autopilot because resources and node pools are explicit.
gcloud container clusters create "$CLUSTER" \
  --region "$REGION" \
  --num-nodes 1 \
  --machine-type e2-standard-16 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 4 || true

gcloud container clusters get-credentials "$CLUSTER" --region "$REGION"

docker build -t "$TRAINER_IMAGE" ../../trainer
docker push "$TRAINER_IMAGE"

docker build -t "$DASHBOARD_IMAGE" ../../dashboard
docker push "$DASHBOARD_IMAGE"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorNamespaceSelectorNilUsesHelmValues=false

kubectl apply -f ../k8s/00-namespace-rbac.yaml
kubectl apply -f ../k8s/10-podmonitor.yaml

TMP=$(mktemp)
sed "s|DASHBOARD_IMAGE_PLACEHOLDER|${DASHBOARD_IMAGE}|g; s|TRAINER_IMAGE_PLACEHOLDER|${TRAINER_IMAGE}|g" \
  ../k8s/20-dashboard-deployment.yaml > "$TMP"
kubectl apply -f "$TMP"
rm "$TMP"

kubectl -n rl get svc rl-dashboard

echo "Done. Wait for EXTERNAL-IP: kubectl -n rl get svc rl-dashboard -w"
