import { NextRequest, NextResponse } from 'next/server';
import * as k8s from '@kubernetes/client-node';

const kc = new k8s.KubeConfig(); kc.loadFromDefault();
const coreApi = kc.makeApiClient(k8s.CoreV1Api);
export async function GET(req: NextRequest, { params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const namespace = process.env.K8S_NAMESPACE || 'rl';
  const pods = await coreApi.listNamespacedPod({ namespace, labelSelector: `job-name=${name}` });
  const pod = pods.items[0];
  if (!pod?.metadata?.name) return NextResponse.json({ logs: 'No pod found yet.' });
  const logs = await coreApi.readNamespacedPodLog({ namespace, name: pod.metadata.name, container: 'trainer', tailLines: 300 });
  return NextResponse.json({ logs });
}
