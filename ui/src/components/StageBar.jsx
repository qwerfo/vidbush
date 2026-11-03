import React from "react";
import { STAGES } from "../format.js";

// 13 stage segments coloured by status, with the running one pulsing.
export default function StageBar({ project }) {
  const order = project.stage_order || STAGES;
  return (
    <ol className="stagebar" aria-label="Pipeline stages">
      {order.map((name) => {
        const rec = (project.stages || {})[name] || { status: "pending" };
        const secs = rec.started_at && rec.finished_at ? `${(rec.finished_at - rec.started_at).toFixed(1)} s` : "";
        const title = `${name}: ${rec.status}${secs ? `, ${secs}` : ""}${rec.error ? `\n${rec.error}` : ""}`;
        return (
          <li key={name} className={`stage stage-${rec.status}`} title={title}>
            <span className="stage-dot" />
            <span className="stage-name">{name}</span>
          </li>
        );
      })}
    </ol>
  );
}
