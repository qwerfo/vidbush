import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { go } from "../hooks.js";

const EMPTY = {
  name: "", voice: { provider: "elevenlabs", voice_id: "", model: "auto", stability: 0.45, similarity: 0.8 },
  style_profile_id: "documentary_default", music_folder: "fixtures/music", intro_path: null, outro_path: null,
  blocklist: [], niche_notes: "", language: "en",
};

function BrandForm({ id, data, profiles, onSaved }) {
  const [b, setB] = useState(data);
  const [block, setBlock] = useState((data.blocklist || []).join("\n"));
  const [msg, setMsg] = useState(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setB(data); setBlock((data.blocklist || []).join("\n")); setMsg(null); }, [data]);

  const set = (k) => (e) => setB({ ...b, [k]: e.target.value });
  const setVoice = (k, num) => (e) => setB({ ...b, voice: { ...b.voice, [k]: num ? Number(e.target.value) : e.target.value } });
  const setPath = (k) => (e) => setB({ ...b, [k]: e.target.value.trim() || null });

  async function save(e) {
    e.preventDefault();
    setSaving(true);
    setMsg(null);
    try {
      const blocklist = block.split(/[\n,]/).map((s) => s.trim()).filter(Boolean);
      const saved = await api.saveBrand(id, { ...b, blocklist });
      setMsg({ ok: true, text: `Saved config/brands/${id}.yaml` });
      onSaved(saved);
    } catch (err) {
      setMsg({ ok: false, text: err.message });
    }
    setSaving(false);
  }

  return (
    <form className="card brand-form" onSubmit={save}>
      <h2>{b.name || id} <span className="muted small">{id}</span></h2>
      <div className="grid2">
        <label>Name<input value={b.name} onChange={set("name")} /></label>
        <label>Language<input value={b.language} onChange={set("language")} placeholder="en" /></label>
      </div>
      <fieldset>
        <legend>Voice</legend>
        <div className="grid4">
          <label>Provider
            <input list="voice-providers" value={b.voice.provider} onChange={setVoice("provider")} />
            <datalist id="voice-providers"><option value="elevenlabs" /><option value="mock" /></datalist>
          </label>
          <label>Voice id<input value={b.voice.voice_id} onChange={setVoice("voice_id")} /></label>
          <label>Model<input value={b.voice.model} onChange={setVoice("model")} placeholder="auto" /></label>
          <label>Stability<input type="number" min={0} max={1} step={0.05} value={b.voice.stability} onChange={setVoice("stability", true)} /></label>
        </div>
      </fieldset>
      <div className="grid2">
        <label>Style profile
          <select value={b.style_profile_id} onChange={set("style_profile_id")}>
            {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}{p.learned ? " (learned)" : ""}</option>)}
          </select>
        </label>
        <label>Music folder<input value={b.music_folder} onChange={set("music_folder")} /></label>
        <label>Intro clip path<input value={b.intro_path || ""} onChange={setPath("intro_path")} placeholder="none" /></label>
        <label>Outro clip path<input value={b.outro_path || ""} onChange={setPath("outro_path")} placeholder="none" /></label>
      </div>
      <label>Blocklist (one domain or term per line)
        <textarea rows={3} value={block} onChange={(e) => setBlock(e.target.value)} />
      </label>
      <label>Niche notes (tone, audience, house rules for the writer)
        <textarea rows={4} value={b.niche_notes} onChange={set("niche_notes")} />
      </label>
      {msg && <div className={msg.ok ? "banner ok" : "error"}>{msg.text}</div>}
      <div className="actions"><button className="btn primary" disabled={saving}>{saving ? "Saving" : "Save brand"}</button></div>
    </form>
  );
}

export default function Brands({ initial }) {
  const [brands, setBrands] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [current, setCurrent] = useState(initial || "default");
  const [data, setData] = useState(null);
  const [newId, setNewId] = useState("");
  const [error, setError] = useState(null);

  const loadList = () => api.brands().then(setBrands).catch((e) => setError(e.message));
  useEffect(() => { loadList(); api.styleProfiles().then(setProfiles).catch(() => {}); }, []);
  useEffect(() => {
    if (initial && initial !== current) setCurrent(initial);
  }, [initial]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setData(null);
    api.brand(current).then(setData).catch(() => setData({ ...EMPTY, id: current, name: current }));
  }, [current]);

  function create(e) {
    e.preventDefault();
    const idv = newId.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(idv)) { setError("Brand id: letters, digits, _ and - only"); return; }
    setError(null);
    setNewId("");
    go(`brands/${idv}`);
    setCurrent(idv);
  }

  return (
    <div className="page">
      <div className="page-head"><h1>Brands</h1></div>
      {error && <div className="error">{error}</div>}
      <div className="split">
        <nav className="card side-list" aria-label="Brands">
          {brands.map((b) => (
            <a key={b.id} href={`#/brands/${b.id}`} className={b.id === current ? "active" : ""} onClick={() => setCurrent(b.id)}>
              {b.name}<span className="muted small">{b.id}</span>
            </a>
          ))}
          <form onSubmit={create} className="new-brand">
            <input value={newId} onChange={(e) => setNewId(e.target.value)} placeholder="new_brand_id" aria-label="New brand id" />
            <button className="btn">Add</button>
          </form>
        </nav>
        {data && <BrandForm id={current} data={data} profiles={profiles} onSaved={(s) => { setData(s); loadList(); }} />}
      </div>
    </div>
  );
}
