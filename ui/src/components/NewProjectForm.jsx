import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { go } from "../hooks.js";

const TABS = [
  ["brief", "Brief"],
  ["script", "Script"],
  ["voiceover", "Voiceover"],
];

// Create a project from a brief, a script or a voiceover, then open it.
export default function NewProjectForm({ onCancel }) {
  const [mode, setMode] = useState("brief");
  const [brands, setBrands] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [f, setF] = useState({
    topic: "", target_minutes: 8, format: "documentary", audience: "", angle: "",
    brand: "default", style_profile: "", script_text: "", project_id: "",
  });
  const [scriptFile, setScriptFile] = useState(null);
  const [voFile, setVoFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.brands().then(setBrands).catch(() => {});
    api.styleProfiles().then(setProfiles).catch(() => {});
  }, []);

  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("mode", mode);
    for (const k of ["brand", "format", "target_minutes", "style_profile", "project_id"]) {
      if (f[k] !== "" && f[k] !== null) form.append(k, f[k]);
    }
    if (mode === "brief") {
      form.append("topic", f.topic);
      form.append("audience", f.audience);
      form.append("angle", f.angle);
    } else {
      if (scriptFile) form.append("script", scriptFile);
      else if (f.script_text.trim()) form.append("script_text", f.script_text);
    }
    if (mode === "voiceover" && voFile) form.append("voiceover", voFile);
    try {
      const res = await api.createProject(form);
      go(`p/${res.project.id}`);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const words = f.script_text.trim() ? f.script_text.trim().split(/\s+/).length : 0;
  return (
    <form className="card new-project" onSubmit={submit}>
      <div className="card-head">
        <h2>New video</h2>
        <button type="button" className="btn ghost" onClick={onCancel}>Close</button>
      </div>
      <div className="tabs" role="tablist">
        {TABS.map(([k, label]) => (
          <button key={k} type="button" role="tab" aria-selected={mode === k}
            className={mode === k ? "tab active" : "tab"} onClick={() => setMode(k)}>{label}</button>
        ))}
      </div>

      {mode === "brief" && (
        <div className="grid2">
          <label className="span2">Topic
            <input autoFocus required value={f.topic} onChange={set("topic")} placeholder="The Panama Canal" />
          </label>
          <label className="span2">Audience notes
            <textarea rows={2} value={f.audience} onChange={set("audience")}
              placeholder="Who is watching, what they already know, tone" />
          </label>
          <label className="span2">Angle
            <input value={f.angle} onChange={set("angle")} placeholder="The human cost of building it" />
          </label>
        </div>
      )}

      {mode !== "brief" && (
        <div className="grid2">
          {mode === "voiceover" && (
            <label className="span2">Voiceover audio (mp3, wav, m4a, flac, ogg)
              <input type="file" required accept=".mp3,.wav,.m4a,.flac,.ogg,audio/*"
                onChange={(e) => setVoFile(e.target.files[0] || null)} />
            </label>
          )}
          <label className="span2">
            {mode === "voiceover" ? "Script for the voiceover (optional, improves alignment)" : "Paste the script"}
            <textarea rows={10} value={f.script_text} onChange={set("script_text")} disabled={Boolean(scriptFile)}
              placeholder={"# Title\n\n## Scene one\nNarration text..."} />
            <span className="hint">{words ? `${words} words, about ${(words / 150).toFixed(1)} min` : ""}</span>
          </label>
          <label className="span2">or upload a script file (.md, .txt)
            <input type="file" accept=".md,.txt,text/plain,text/markdown"
              onChange={(e) => setScriptFile(e.target.files[0] || null)} />
          </label>
        </div>
      )}

      <div className="grid4">
        <label>Brand
          <select value={f.brand} onChange={set("brand")}>
            {(brands.length ? brands : [{ id: "default", name: "Default brand" }]).map((b) => (
              <option key={b.id} value={b.id}>{b.name}</option>
            ))}
          </select>
        </label>
        <label>Format
          <select value={f.format} onChange={set("format")}>
            <option value="documentary">Documentary</option>
            <option value="top10">Top 10</option>
          </select>
        </label>
        <label>Target minutes
          <input type="number" min={1} max={90} step={1} value={f.target_minutes} onChange={set("target_minutes")} />
        </label>
        <label>Style profile
          <select value={f.style_profile} onChange={set("style_profile")}>
            <option value="">Brand default</option>
            {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </label>
      </div>
      <details className="advanced">
        <summary>Advanced</summary>
        <label>Project id (optional, letters, digits, _ and -)
          <input value={f.project_id} onChange={set("project_id")} pattern="[A-Za-z0-9][A-Za-z0-9_\-]*" />
        </label>
      </details>
      {error && <div className="error">{error}</div>}
      <div className="actions">
        <button className="btn primary" disabled={busy}>{busy ? "Creating" : "Create and start"}</button>
      </div>
    </form>
  );
}
