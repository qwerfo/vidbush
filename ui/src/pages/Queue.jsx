import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { ago } from "../format.js";
import { go, usePoll } from "../hooks.js";
import StatusPill from "../components/StatusPill.jsx";

const LIVE = ["queued", "running"];

function money(v) {
  return v === null || v === undefined ? "-" : `$${Number(v).toFixed(2)}`;
}

function mmss(sec) {
  if (sec === null || sec === undefined) return "-";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

// Queue a folder or list file of briefs, scripts and voiceovers (paths on the server).
function NewBatchForm({ onDone, onCancel }) {
  const [brands, setBrands] = useState([]);
  const [f, setF] = useState({ source: "", brand: "default", cost_cap_usd: "", id_prefix: "", format: "", minutes: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.brands().then(setBrands).catch(() => {});
  }, []);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const body = { source: f.source.trim(), brand: f.brand, id_prefix: f.id_prefix.trim() };
    if (f.cost_cap_usd !== "") body.cost_cap_usd = Number(f.cost_cap_usd);
    if (f.format) body.format = f.format;
    if (f.minutes !== "") body.minutes = Number(f.minutes);
    try {
      const res = await api.createBatch(body);
      onDone(res.batch_id);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  return (
    <form className="card" onSubmit={submit}>
      <h2>New batch</h2>
      <p className="hint">A folder (every brief, script and voiceover in it, sorted by name) or a list file with one path per line. Paths are on the machine that runs Vidbush.</p>
      <div className="grid2">
        <label className="span2">Folder or list file
          <input required value={f.source} onChange={set("source")} placeholder="C:\vidbush\briefs or /app/briefs/tonight.txt" />
        </label>
        <label>Brand
          <select value={f.brand} onChange={set("brand")}>
            {(brands.length ? brands : [{ id: "default", name: "default" }]).map((b) => (
              <option key={b.id} value={b.id}>{b.name}</option>
            ))}
          </select>
        </label>
        <label>Cost cap per project (USD)
          <input type="number" min="0" step="0.5" value={f.cost_cap_usd} onChange={set("cost_cap_usd")} placeholder="config default, 0 = no cap" />
        </label>
        <label>Format
          <select value={f.format} onChange={set("format")}>
            <option value="">from each brief</option>
            <option value="documentary">Documentary</option>
            <option value="top10">Top 10</option>
          </select>
        </label>
        <label>Minutes
          <input type="number" min="1" max="90" value={f.minutes} onChange={set("minutes")} placeholder="from each brief" />
        </label>
        <label>Project id prefix
          <input value={f.id_prefix} onChange={set("id_prefix")} placeholder="optional" />
        </label>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="actions">
        <button type="button" className="btn ghost" onClick={onCancel}>Cancel</button>
        <button className="btn primary" disabled={busy || !f.source.trim()}>{busy ? "Queueing" : "Queue and start"}</button>
      </div>
    </form>
  );
}

function BatchDetail({ id, onChange }) {
  const [interval, setIntervalMs] = useState(3000);
  const [err, setErr] = useState(null);
  const [b, error, reload] = usePoll(async () => {
    const r = await api.batch(id);
    setIntervalMs(LIVE.includes(r.status) ? 3000 : 15000);
    return r;
  }, [id], interval);

  async function act(fn) {
    setErr(null);
    try {
      await fn(id);
      await reload();
      onChange();
    } catch (e) {
      setErr(e.message);
    }
  }

  if (error) return <div className="error">{error}</div>;
  if (!b) return <div className="muted">Loading</div>;
  const s = b.summary;
  const unfinished = b.rows.some((r) => LIVE.includes(r.status));
  return (
    <section>
      <div className="card-head">
        <h2>
          Batch {b.batch_id} <StatusPill status={b.status} />
          {b.paused && b.status === "running" && <span className="pill pill-paused">pausing after this project</span>}
        </h2>
        <div className="head-actions">
          <button className="btn" disabled={b.paused || !unfinished} onClick={() => act(api.pauseBatch)}>Pause</button>
          <button className="btn primary" disabled={!unfinished || (!b.paused && b.status !== "interrupted")}
            onClick={() => act(api.resumeBatch)}>Resume</button>
        </div>
      </div>
      {err && <div className="error">{err}</div>}
      <p className="small muted">
        <code>{b.source}</code> for brand {b.brand}. Cost cap {b.cost_cap_usd ? money(b.cost_cap_usd) : "none"} per project.
        Total {money(s.total_cost_usd)} for {s.total_minutes} min ({money(s.cost_per_min_usd)} per minute).
        Pause takes effect between projects. Report: <code>{b.report_path}</code>
      </p>
      <div className="table-wrap"><table className="table queue">
        <thead>
          <tr><th>Project</th><th>Input</th><th>Mode</th><th>Status</th><th>Duration</th><th>Cost</th><th>Cost/min</th><th>QA</th><th>Note</th></tr>
        </thead>
        <tbody>
          {b.rows.map((r, i) => (
            <tr key={`${r.project_id}-${i}`}>
              <td>{r.status === "skipped" || r.status === "queued" ? r.project_id : <a href={`#/p/${encodeURIComponent(r.project_id)}`}>{r.project_id}</a>}</td>
              <td className="small" title={r.input}>{r.input.split(/[\\/]/).pop()}</td>
              <td>{r.mode || "-"}</td>
              <td><StatusPill status={r.status} /></td>
              <td>{mmss(r.duration_s)}</td>
              <td>{money(r.cost_usd)}</td>
              <td>{money(r.cost_per_min_usd)}</td>
              <td>{r.qa_pct === null || r.qa_pct === undefined ? "-" : `${r.qa_pct}%`}</td>
              <td className="small muted note">{r.note}</td>
            </tr>
          ))}
        </tbody>
      </table></div>
    </section>
  );
}

export default function Queue({ initial }) {
  const [creating, setCreating] = useState(false);
  const [list, error, reload] = usePoll(() => api.batches(), [], 5000);
  const current = initial || (list && list[0] && list[0].id);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Queue</h1>
        {!creating && <button className="btn primary" onClick={() => setCreating(true)}>New batch</button>}
      </div>
      {creating && (
        <NewBatchForm onCancel={() => setCreating(false)}
          onDone={(bid) => { setCreating(false); reload(); go(`queue/${bid}`); }} />
      )}
      {error && <div className="error">{error}</div>}
      {list && list.length === 0 && !creating && (
        <div className="empty card">
          <p>No batches yet. Queue a folder of briefs, scripts or voiceovers to render them one after another.</p>
          <p className="small muted">Same from the command line: <code>python -m vidbush batch briefs --brand default</code></p>
          <button className="btn primary" onClick={() => setCreating(true)}>New batch</button>
        </div>
      )}
      {list && list.length > 0 && (
        <div className="split">
          <nav className="card side-list" aria-label="Batches">
            {list.map((b) => (
              <a key={b.id} href={`#/queue/${encodeURIComponent(b.id)}`} className={b.id === current ? "active" : ""}>
                <span>{b.id} <StatusPill status={b.status} /></span>
                <span className="muted small">{b.items} items, {money(b.total_cost_usd)}, {ago(b.created_at)}</span>
              </a>
            ))}
          </nav>
          <div className="card">{current && <BatchDetail id={current} key={current} onChange={reload} />}</div>
        </div>
      )}
    </div>
  );
}
