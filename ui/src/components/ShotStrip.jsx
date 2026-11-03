import React, { useEffect, useRef } from "react";
import { TYPE_COLORS, TYPE_LABELS, clock, score } from "../format.js";

// Horizontal strip of shot thumbnails. Width is proportional to duration (clamped).
export default function ShotStrip({ shots, selectedId, playingId, threshold, onPick }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current && ref.current.querySelector(`[data-shot="${playingId}"]`);
    if (el) el.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  }, [playingId]);
  return (
    <div className="strip" ref={ref} role="listbox" aria-label="Shots">
      {shots.map((s) => {
        const low = s.judge_score !== null && s.judge_score !== undefined && s.judge_score < threshold;
        const flagged = (s.flags || []).length > 0 || low;
        const cls = ["shot", s.id === selectedId && "selected", s.id === playingId && "playing"].filter(Boolean).join(" ");
        return (
          <button key={s.id} data-shot={s.id} className={cls} role="option" aria-selected={s.id === selectedId}
            style={{ width: `${Math.min(220, Math.max(96, s.duration * 18))}px` }}
            title={`#${s.index + 1} ${TYPE_LABELS[s.type] || s.type} ${clock(s.start)} to ${clock(s.end)}\n${s.intent}`}
            onClick={() => onPick(s)}>
            <img src={s.thumb_url} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
            <span className="shot-type" style={{ background: TYPE_COLORS[s.type] }}>{TYPE_LABELS[s.type] || s.type}</span>
            <span className={`shot-score ${low ? "low" : ""}`}>{score(s.judge_score)}</span>
            {flagged && <span className="shot-flag" title={(s.flags || []).join(", ") || "below threshold"}>!</span>}
            {s.overlay && <span className="shot-ov" title={`overlay: ${s.overlay}`}>OV</span>}
            <span className="shot-time">{clock(s.start)}</span>
          </button>
        );
      })}
    </div>
  );
}
