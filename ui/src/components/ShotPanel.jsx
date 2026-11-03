import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { TYPE_COLORS, TYPE_LABELS, clock, score } from "../format.js";

function Preview({ cand, fallback }) {
  if (!cand) return <img className="preview" src={fallback} alt="Shot preview" />;
  if (cand.media === "video") {
    return <video className="preview" src={cand.media_url} poster={cand.thumb_url} controls muted loop playsInline />;
  }
  return <img className="preview" src={cand.thumb_url} alt="Selected visual" />;
}

// Details and actions for one shot. Every action enqueues an apply_ops job via onJob(jobId).
export default function ShotPanel({ projectId, shot, busy, onJob, onClose }) {
  const [prompt, setPrompt] = useState("");
  const [secs, setSecs] = useState(shot.duration);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);
  useEffect(() => {
    setSecs(shot.duration);
    setError(null);
  }, [shot.id, shot.duration]);

  const selected = shot.candidates.find((c) => c.selected);
  const others = shot.candidates.filter((c) => !c.selected);

  async function act(fn, label) {
    setError(null);
    try {
      const res = await fn();
      onJob(res.job_id, label);
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <aside className="card panel" aria-label="Shot details">
      <div className="card-head">
        <h3>Shot {shot.index + 1}
          <span className="tag" style={{ background: TYPE_COLORS[shot.type] }}>{TYPE_LABELS[shot.type] || shot.type}</span>
        </h3>
        <button className="btn ghost" onClick={onClose} title="Close (Esc)">Close</button>
      </div>
      <p className="intent">{shot.intent}</p>
      <dl className="facts">
        <dt>Timing</dt><dd>{clock(shot.start)} to {clock(shot.end)} ({shot.duration.toFixed(2)} s)</dd>
        <dt>Judge</dt><dd>{score(shot.judge_score)}{shot.attempts ? `, ${shot.attempts} attempt(s)` : ""}</dd>
        {shot.judge_notes && (<><dt>Notes</dt><dd>{shot.judge_notes}</dd></>)}
        {shot.overlay && (<><dt>Overlay</dt><dd>{shot.overlay}</dd></>)}
        {shot.motion && (<><dt>Graphic</dt><dd>{shot.motion}</dd></>)}
        {(shot.flags || []).length > 0 && (<><dt>Flags</dt><dd className="warn">{shot.flags.join(", ")}</dd></>)}
        {selected && (<><dt>Source</dt><dd>{selected.source}, {selected.licence}{selected.author ? `, ${selected.author}` : ""}</dd></>)}
      </dl>
      <Preview cand={selected} fallback={shot.thumb_url} />

      <div className="actions-grid">
        <div className="action">
          <button className="btn" disabled={busy} onClick={() => fileRef.current && fileRef.current.click()}>Upload my own</button>
          <input ref={fileRef} type="file" hidden accept="image/*,video/*"
            onChange={(e) => {
              const f = e.target.files[0];
              e.target.value = "";
              if (f) act(() => api.upload(projectId, shot.id, f), `Uploading ${f.name}`);
            }} />
        </div>
        <form className="action" onSubmit={(e) => { e.preventDefault(); act(() => api.regenerate(projectId, shot.id, prompt), "Generating an AI image"); }}>
          <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Optional prompt tweak, e.g. at dusk, wide angle" />
          <button className="btn" disabled={busy}>Regenerate as AI image</button>
        </form>
        <form className="action inline" onSubmit={(e) => { e.preventDefault(); act(() => api.duration(projectId, shot.id, Number(secs)), "Changing duration"); }}>
          <label className="inline-label">Duration (s)
            <input type="number" min={0.6} max={60} step={0.1} value={secs} onChange={(e) => setSecs(e.target.value)} />
          </label>
          <button className="btn" disabled={busy || Number(secs) === shot.duration}>Edit duration</button>
        </form>
        {(shot.overlay || shot.overlay_url) && (
          <div className="action">
            <button className="btn danger" disabled={busy} onClick={() => act(() => api.removeOverlay(projectId, shot.id), "Removing overlay")}>Remove overlay</button>
          </div>
        )}
      </div>
      {error && <div className="error">{error}</div>}

      <h4>Other candidates ({others.length})</h4>
      {others.length === 0 && <p className="muted small">No other candidates for this shot.</p>}
      <div className="cands">
        {others.map((c) => (
          <div key={c.id} className={`cand ${c.reject_reason ? "rejected" : ""}`}>
            <a href={c.media_url} target="_blank" rel="noreferrer" title="Open full size">
              <img src={c.thumb_url} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
            </a>
            <div className="cand-meta">
              <span className="cand-score">{score(c.score)}</span>
              <span className="small">{c.source}{c.media === "video" ? " video" : ""}</span>
              <span className="small muted" title={c.licence}>{c.licence}</span>
            </div>
            {c.reject_reason && <div className="small reject" title={c.reject_reason}>{c.reject_reason}</div>}
            {!c.reject_reason && c.judge_reason && <div className="small muted clamp" title={c.judge_reason}>{c.judge_reason}</div>}
            <button className="btn small-btn" disabled={busy} onClick={() => act(() => api.swap(projectId, shot.id, c.id), "Swapping shot")}>Use this one</button>
          </div>
        ))}
      </div>
    </aside>
  );
}
