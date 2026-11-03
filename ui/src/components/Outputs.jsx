import React from "react";

const LABELS = {
  "description.md": "Description",
  "chapters.txt": "Chapters",
  "captions.srt": "Captions (SRT)",
  "captions.ass": "Captions (ASS)",
  "qa_report.md": "QA report",
  "cost_report.json": "Cost report",
  "shotlist.json": "Shot list",
};

export default function Outputs({ outputs, finalUrl }) {
  if (!outputs) return null;
  const stats = outputs.render_stats;
  return (
    <div className="card outputs">
      <h3>Deliverables</h3>
      <div className="links">
        {finalUrl && <a className="btn" href={finalUrl} download="final.mp4">Download final.mp4</a>}
        {Object.entries(LABELS).filter(([k]) => outputs[k]).map(([k, label]) => (
          <a key={k} className="btn ghost" href={outputs[k]} target="_blank" rel="noreferrer">{label}</a>
        ))}
      </div>
      {(outputs.thumbnails || []).length > 0 && (
        <div className="thumbs">
          {outputs.thumbnails.map((t) => (
            <a key={t} href={t} target="_blank" rel="noreferrer"><img src={t} alt="Thumbnail option" /></a>
          ))}
        </div>
      )}
      {stats && (
        <p className="small muted">Last assembly: {stats.rendered} segments rendered, {stats.cached} from cache, {stats.segments} total.</p>
      )}
    </div>
  );
}
