import React, { useState } from "react";
import { api } from "../api.js";
import { go, usePoll } from "../hooks.js";
import { ago } from "../format.js";
import StatusPill from "../components/StatusPill.jsx";
import NewProjectForm from "../components/NewProjectForm.jsx";

export default function Projects() {
  const [creating, setCreating] = useState(false);
  const [interval, setIntervalMs] = useState(3000);
  const [rows, error] = usePoll(async () => {
    const r = await api.projects();
    setIntervalMs(r.some((p) => p.status === "running" || p.active_job) ? 3000 : 15000);
    return r;
  }, [], interval);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Projects</h1>
        {!creating && <button className="btn primary" onClick={() => setCreating(true)}>New video</button>}
      </div>
      {creating && <NewProjectForm onCancel={() => setCreating(false)} />}
      {error && <div className="error">{error}</div>}
      {rows && rows.length === 0 && !creating && (
        <div className="empty card">
          <p>No projects yet.</p>
          <button className="btn primary" onClick={() => setCreating(true)}>Make your first video</button>
        </div>
      )}
      {rows && rows.length > 0 && (
        <table className="table projects">
          <thead>
            <tr><th>Title</th><th>Input</th><th>Format</th><th>Status</th><th>Stage</th><th>Created</th></tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.id} tabIndex={0} onClick={() => go(`p/${p.id}`)}
                onKeyDown={(e) => e.key === "Enter" && go(`p/${p.id}`)}>
                <td>
                  <a href={`#/p/${encodeURIComponent(p.id)}`} onClick={(e) => e.stopPropagation()}>{p.title || p.id}</a>
                  <div className="muted small">{p.id}</div>
                </td>
                <td>{p.input_mode}</td>
                <td>{p.format === "top10" ? "Top 10" : "Documentary"}</td>
                <td><StatusPill status={p.active_job && p.status !== "running" ? "queued" : p.status}
                  label={p.active_job && p.active_job.kind === "apply_ops" ? "re-rendering" : undefined} /></td>
                <td>
                  <div className="mini-progress" title={`${p.stages_done || 0} of 13 stages`}>
                    <span style={{ width: `${((p.stages_done || 0) / 13) * 100}%` }} />
                  </div>
                  <span className="small muted">{p.current_stage || (p.has_final ? "final.mp4 ready" : "")}</span>
                </td>
                <td className="muted">{ago(p.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
