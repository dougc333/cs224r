import { NextResponse } from 'next/server';
import * as k8s from '@kubernetes/client-node';

const kc = new k8s.KubeConfig();
kc.loadFromDefault();
const batchApi = kc.makeApiClient(k8s.BatchV1Api);
const coreApi = kc.makeApiClient(k8s.CoreV1Api);

export async function GET() {
  const namespace = process.env.K8S_NAMESPACE || 'rl';
  const jobsRes = await batchApi.listNamespacedJob({ namespace, labelSelector: 'app=mujoco-rl' });
  const podsRes = await coreApi.listNamespacedPod({ namespace, labelSelector: 'app=mujoco-rl' });
  const podByJob = new Map<string, any>();
  for (const pod of podsRes.items) {
    const jobName = pod.metadata?.labels?.['job-name'];
    if (jobName) podByJob.set(jobName, pod);
  }
  const jobs = jobsRes.items.map((job) => {
    const name = job.metadata?.name || '';
    const pod = podByJob.get(name);
    return {
      name,
      labels: job.metadata?.labels || {},
      active: job.status?.active || 0,
      succeeded: job.status?.succeeded || 0,
      failed: job.status?.failed || 0,
      startTime: job.status?.startTime,
      completionTime: job.status?.completionTime,
      podName: pod?.metadata?.name || '',
      podPhase: pod?.status?.phase || 'Pending',
    };
  });
  return NextResponse.json({ jobs });
}
