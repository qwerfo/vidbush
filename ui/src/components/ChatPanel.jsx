import React, { useState } from "react";
import { api } from "../api.js";
import { opSummary } from "../format.js";

// Plain-language revisions: map the message to EditOps, show them, apply on confirm.
export default function ChatPanel({ projectId, busy, onJob }) {
  const [msg, setMsg] = useState("");
  const [ops, setOps] = useState(null);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState(null);

  async function preview(e) {
    e.preventDefault();
    if (!msg.trim()) return;
    setThinking(true);
    setError(null);
    try {
      const res = await api.chat(projectId, msg.trim(), false);
      setOps(res.ops);
    } catch (err) {
      setError(err.message);
    }
    setThinking(false);
  }

  async function apply() {
    setError(null);
    try {
      const res = await api.ops(projectId, ops);
      onJob(res.job_id, `Applying ${ops.length} edit(s)`);
      setOps(null);
      setMsg("");
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <aside className="card panel" aria-label="Chat revisions">
      <h3>Revise with chat</h3>
      <form onSubmit={preview}>
        <textarea rows={4} value={msg} onChange={(e) => setMsg(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) preview(e); }}
          placeholder={"e.g. Make shot 4 two seconds longer\nUse calmer music\nCaptions in yellow"} />
        <div className="actions">
          <span className="hint">Ctrl+Enter to send</span>
          <button className="btn" disabled={thinking || !msg.trim()}>{thinking ? "Thinking" : "Preview edits"}</button>
        </div>
      </form>
      {error && <div className="error">{error}</div>}
      {ops && ops.length === 0 && <p className="muted">No edits matched that request. Try naming a shot number or a time.</p>}
      {ops && ops.length > 0 && (
        <>
          <ol className="ops">
            {ops.map((o, i) => (
              <li key={i}>
                <code>{o.op}</code> <span className="small">{opSummary(o)}</span>
                {o.reason && <div className="small muted">{o.reason}</div>}
              </li>
            ))}
          </ol>
          <div className="actions">
            <button className="btn ghost" onClick={() => setOps(null)}>Discard</button>
            <button className="btn primary" disabled={busy} onClick={apply}>Apply {ops.length} edit(s)</button>
          </div>
        </>
      )}
    </aside>
  );
}
