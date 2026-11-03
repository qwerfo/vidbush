import React from "react";
import { api } from "../api.js";
import { usePoll } from "../hooks.js";

function Check({ ok }) {
  return <span className={`pill ${ok ? "pill-done" : "pill-failed"}`}>{ok ? "ok" : "missing"}</span>;
}

export default function SettingsPage() {
  const [s, error, reload] = usePoll(() => api.settings(), [], 0);
  if (error) return <div className="page"><div className="error">{error}</div></div>;
  if (!s) return <div className="page muted">Loading</div>;
  const d = s.doctor;
  return (
    <div className="page">
      <div className="page-head">
        <h1>Settings</h1>
        <button className="btn" onClick={reload}>Re-check</button>
      </div>
      {s.offline && <div className="banner warnbox">Offline mode is on: every seat answers from the mock provider and no network is used.</div>}
      <div className="split2">
        <section className="card">
          <h2>Seats</h2>
          <p className="muted small">A seat is a job role. Real means at least one provider for the role has its key set in .env. Mock seats use canned fixtures and placeholders.</p>
          <table className="table">
            <thead><tr><th>Role</th><th>Resolves to</th><th>Providers (in order)</th></tr></thead>
            <tbody>
              {Object.entries(s.seats).map(([role, v]) => (
                <tr key={role}>
                  <td><code>{role}</code></td>
                  <td><span className={`pill ${v.real ? "pill-done" : "pill-mock"}`}>{v.real ? "real" : "mock"}</span></td>
                  <td className="small">
                    {v.providers.length === 0 && <span className="muted">none configured</span>}
                    {v.providers.map((p, i) => (
                      <span key={i} className={p.configured ? "" : "muted"} title={p.configured ? "key present" : "no key"}>
                        {i > 0 && ", "}{p.name}{p.model ? ` (${p.model})` : ""}{p.configured ? "" : " [no key]"}
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="card">
          <h2>Doctor <Check ok={d.ok} /></h2>
          <h4>Binaries</h4>
          <ul className="checks">{Object.entries(d.binaries).map(([k, v]) => <li key={k}><Check ok={v} /> <code>{k}</code></li>)}</ul>
          <h4>Python modules</h4>
          <ul className="checks">{Object.entries(d.python_modules).map(([k, v]) => <li key={k}><Check ok={v} /> <code>{k}</code></li>)}</ul>
          <h4>Fonts</h4>
          <p className="small">{d.fonts.length ? d.fonts.join(", ") : <span className="warn">No fonts found in assets/fonts</span>}</p>
          <h4>Sources</h4>
          <p className="small">Enabled: {s.sources.enabled.join(", ")}. YouTube {s.sources.allow_youtube ? "allowed" : "off"}. Up to {s.sources.max_candidates} candidates per shot. Local library: <code>{s.sources.local_library}</code></p>
          <h4>Render</h4>
          <p className="small">{s.render.width}x{s.render.height} at {s.render.fps} fps, renderer {s.render.renderer}, preset {s.render.preset}, crf {s.render.crf}, {s.render.workers} workers, target {s.render.target_lufs} LUFS.</p>
          <h4>Judge</h4>
          <p className="small">Threshold {s.judge.threshold}, up to {s.judge.max_attempts} attempts per shot.</p>
        </section>
      </div>
    </div>
  );
}
