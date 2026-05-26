import { NextRequest, NextResponse } from 'next/server';
const PROM_URL = process.env.PROMETHEUS_URL || 'http://prometheus-kube-prometheus-prometheus.monitoring:9090';
export async function GET(req: NextRequest) {
  const p = req.nextUrl.searchParams;
  const query = p.get('query');
  const start = p.get('start');
  const end = p.get('end');
  const step = p.get('step') || '15s';
  if (!query || !start || !end) return NextResponse.json({ error: 'missing query/start/end' }, { status: 400 });
  const url = `${PROM_URL}/api/v1/query_range?query=${encodeURIComponent(query)}&start=${start}&end=${end}&step=${step}`;
  const res = await fetch(url, { cache: 'no-store' });
  return NextResponse.json(await res.json());
}
