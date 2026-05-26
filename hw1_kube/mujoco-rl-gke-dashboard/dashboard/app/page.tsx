'use client';

import { useEffect, useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts';

const ENVS = ['Ant-v5','Humanoid-v5','HalfCheetah-v5','Walker2d-v5','Hopper-v5','Swimmer-v5','Reacher-v5','Pusher-v5'];
const ALGOS = ['PPO','SAC','TD3','A2C'];
const METRICS = [
  { key: 'rl_episode_return_mean', label: 'reward mean' },
  { key: 'rl_episode_return_min', label: 'reward min' },
  { key: 'rl_episode_length_mean', label: 'episode len mean' },
  { key: 'rl_episode_length_min', label: 'episode len min' },
  { key: 'rl_training_steps', label: 'steps' },
  { key: 'rl_train_loss', label: 'loss' },
];

type Job = { name:string; labels:any; active:number; succeeded:number; failed:number; podPhase:string; startTime:string; completionTime:string };

function statusOf(j: Job) {
  if (j.succeeded) return 'Succeeded';
  if (j.failed) return 'Failed';
  if (j.active) return 'Running';
  return j.podPhase || 'Pending';
}

function MetricChart({ runId, metric, title }: { runId:string; metric:string; title:string }) {
  const [data, setData] = useState<any[]>([]);
  useEffect(() => {
    async function load() {
      const end = Math.floor(Date.now()/1000);
      const start = end - 60*60*3;
      const q = metric === 'rl_train_loss'
        ? `${metric}{run_id="${runId}"}`
        : `${metric}{run_id="${runId}"}`;
      const r = await fetch(`/api/prometheus/range?query=${encodeURIComponent(q)}&start=${start}&end=${end}&step=15s`, { cache: 'no-store' });
      const json = await r.json();
      const series = json?.data?.result || [];
      if (series.length === 0) { setData([]); return; }
      const rows = new Map<number, any>();
      for (const s of series) {
        const name = s.metric?.name || s.metric?.__name__ || s.metric?.algo || 'value';
        const lossName = s.metric?.name || 'value';
        const label = metric === 'rl_train_loss' ? lossName : 'value';
        for (const [t, v] of s.values) {
          const ti = Number(t);
          const row = rows.get(ti) || { time: new Date(ti*1000).toLocaleTimeString() };
          row[label] = Number(v);
          rows.set(ti, row);
        }
      }
      setData(Array.from(rows.values()));
    }
    load(); const id = setInterval(load, 5000); return () => clearInterval(id);
  }, [runId, metric]);
  const keys = useMemo(() => Array.from(new Set(data.flatMap(d => Object.keys(d).filter(k => k !== 'time')))), [data]);
  return <div className="card" style={{height:320}}><h3>{title}</h3><ResponsiveContainer width="100%" height="85%"><LineChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="time" minTickGap={32}/><YAxis/><Tooltip/><Legend/>{keys.map(k => <Line key={k} type="monotone" dataKey={k} dot={false}/>)}</LineChart></ResponsiveContainer></div>;
}

function Logs({ jobName }: { jobName:string }) {
  const [logs, setLogs] = useState('');
  useEffect(() => { async function load(){ const r=await fetch(`/api/jobs/${jobName}/logs`); const j=await r.json(); setLogs(j.logs || ''); } load(); const id=setInterval(load,5000); return()=>clearInterval(id);}, [jobName]);
  return <pre>{logs}</pre>;
}

export default function Home() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [envId, setEnvId] = useState('Ant-v5');
  const [algo, setAlgo] = useState('PPO');
  const [totalTimesteps, setTotalTimesteps] = useState(1000000);
  const [nEnvs, setNEnvs] = useState(8);
  const [selected, setSelected] = useState<Job | null>(null);
  const [message, setMessage] = useState('');

  async function refresh() { const r = await fetch('/api/jobs', { cache: 'no-store' }); const j = await r.json(); setJobs(j.jobs || []); }
  useEffect(() => { refresh(); const id=setInterval(refresh,5000); return()=>clearInterval(id);}, []);

  async function launch() {
    const r = await fetch('/api/jobs/create', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({envId, algo, totalTimesteps, nEnvs}) });
    const j = await r.json(); setMessage(j.error || `Launched ${j.jobName}`); refresh();
  }

  async function launchAll() {
    for (const e of ENVS) {
      await fetch('/api/jobs/create', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({envId:e, algo, totalTimesteps, nEnvs}) });
    }
    setMessage(`Launched ${ENVS.length} jobs`); refresh();
  }

  const runId = selected?.labels?.run_id;

  return <main>
    <h1>MuJoCo RL Kubernetes Dashboard</h1>
    <p>Launch and monitor Ant, Humanoid, HalfCheetah, Walker2d, Hopper, Swimmer, Reacher, and Pusher jobs.</p>
    <div className="grid">
      <div className="card">
        <h2>Launch Job</h2>
        <p><label>Env<br/><select value={envId} onChange={e=>setEnvId(e.target.value)}>{ENVS.map(e=><option key={e}>{e}</option>)}</select></label></p>
        <p><label>Algo<br/><select value={algo} onChange={e=>setAlgo(e.target.value)}>{ALGOS.map(a=><option key={a}>{a}</option>)}</select></label></p>
        <p><label>Total timesteps<br/><input type="number" value={totalTimesteps} onChange={e=>setTotalTimesteps(Number(e.target.value))}/></label></p>
        <p><label>Parallel envs<br/><input type="number" value={nEnvs} onChange={e=>setNEnvs(Number(e.target.value))}/></label></p>
        <button className="btn" onClick={launch}>Launch selected</button>{' '}
        <button className="btn2" onClick={launchAll}>Launch all 8 envs</button>
        <p>{message}</p>
      </div>
      <div className="card">
        <h2>Metric placement</h2>
        <p>Prometheus should store time-series metrics: reward, episode length, loss, steps, CPU, memory, pod status. React should query and visualize them. Use a DB later for experiment metadata and artifact links.</p>
      </div>
    </div>
    <div className="card" style={{marginTop:16}}>
      <h2>Jobs</h2>
      <table><thead><tr><th>Status</th><th>Env</th><th>Algo</th><th>Run</th><th>Job</th><th>Start</th><th></th></tr></thead><tbody>{jobs.map(j=><tr key={j.name}><td>{statusOf(j)}</td><td>{j.labels?.env}</td><td>{j.labels?.algo}</td><td>{j.labels?.run_id}</td><td>{j.name}</td><td>{j.startTime}</td><td><button className="btn2" onClick={()=>setSelected(j)}>Open</button></td></tr>)}</tbody></table>
    </div>
    {selected && runId && <>
      <h2>Run: {runId}</h2>
      <div className="grid">
        {METRICS.map(m => <MetricChart key={m.key} runId={runId} metric={m.key} title={m.label}/>) }
      </div>
      <h2>Logs</h2><Logs jobName={selected.name}/>
    </>}
  </main>;
}
