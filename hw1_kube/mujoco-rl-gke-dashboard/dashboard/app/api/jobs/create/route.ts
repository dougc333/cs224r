import { NextRequest, NextResponse } from 'next/server';
import * as k8s from '@kubernetes/client-node';

const kc = new k8s.KubeConfig();
kc.loadFromDefault();
const batchApi = kc.makeApiClient(k8s.BatchV1Api);

function sanitize(s: string) { return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''); }

export async function POST(req: NextRequest) {
  const body = await req.json();
  const namespace = process.env.K8S_NAMESPACE || 'rl';
  const trainerImage = process.env.TRAINER_IMAGE;
  if (!trainerImage) return NextResponse.json({ error: 'TRAINER_IMAGE env var missing' }, { status: 500 });

  const envId = body.envId || 'Ant-v5';
  const algo = body.algo || 'PPO';
  const shortId = `${sanitize(envId)}-${sanitize(algo)}-${Date.now()}`;
  const runId = body.runId || shortId;
  const jobName = `rl-${shortId}`.slice(0, 62);

  const envVars = [
    { name: 'RUN_ID', value: runId },
    { name: 'ENV_ID', value: envId },
    { name: 'ALGO', value: algo },
    { name: 'TOTAL_TIMESTEPS', value: String(body.totalTimesteps || 1000000) },
    { name: 'N_ENVS', value: String(body.nEnvs || 8) },
    { name: 'SEED', value: String(body.seed || 42) },
    { name: 'PROMETHEUS_PORT', value: '8000' },
    { name: 'MUJOCO_GL', value: 'osmesa' },
    { name: 'RECORD_VIDEO', value: String(Boolean(body.recordVideo || false)) },
  ];

  const extraNames = ['LEARNING_RATE','N_STEPS','BATCH_SIZE','N_EPOCHS','GAMMA','GAE_LAMBDA','CLIP_RANGE','BUFFER_SIZE','OUTPUT_BUCKET'];
  for (const name of extraNames) {
    const key = name.toLowerCase().replace(/_([a-z])/g, (_, c) => c.toUpperCase());
    if (body[key] !== undefined) envVars.push({ name, value: String(body[key]) });
  }

  const job: any = {
    apiVersion: 'batch/v1',
    kind: 'Job',
    metadata: { name: jobName, namespace, labels: { app: 'mujoco-rl', env: envId, algo, run_id: runId } },
    spec: {
      backoffLimit: 0,
      template: {
        metadata: { labels: { app: 'mujoco-rl', env: envId, algo, run_id: runId } },
        spec: {
          serviceAccountName: 'rl-trainer',
          restartPolicy: 'Never',
          containers: [{
            name: 'trainer', image: trainerImage, imagePullPolicy: 'Always',
            ports: [{ containerPort: 8000, name: 'metrics' }],
            env: envVars,
            resources: {
              requests: { cpu: String(body.cpuRequest || '4'), memory: String(body.memoryRequest || '8Gi') },
              limits: { cpu: String(body.cpuLimit || '8'), memory: String(body.memoryLimit || '16Gi') },
            },
            volumeMounts: [{ name: 'outputs', mountPath: '/outputs' }],
          }],
          volumes: [{ name: 'outputs', emptyDir: {} }],
        },
      },
    },
  };

  await batchApi.createNamespacedJob({ namespace, body: job });
  return NextResponse.json({ jobName, runId });
}
