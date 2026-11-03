import React, { useEffect, useRef } from "react";
import { api } from "../api.js";
import { usePoll } from "../hooks.js";

const SKIP = new Set(["ts", "level", "project", "msg"]);

function fields(rec) {
  return Object.entries(rec)
    .filter(([k]) => !SKIP.has(k))
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`)
    .join(" ");
}

// Tail of the project's JSON-lines log, polled every 2 s while live.
export default function LogTail({ id, live }) {
  const [rows] = usePoll(() => api.log(id, 200), [id], live ? 2000 : 0);
  const box = useRef(null);
  useEffect(() => {
    const el = box.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [rows]);
  return (
    <div className="log" ref={box} aria-live="off">
      {(rows || []).length === 0 && <div className="muted">No log lines yet.</div>}
      {(rows || []).map((r, i) => (
        <div key={i} className={`log-line log-${r.level}`}>
          <span className="log-ts">{new Date(r.ts * 1000).toLocaleTimeString()}</span>
          <span className="log-msg">{r.msg}</span>
          <span className="log-fields">{fields(r)}</span>
        </div>
      ))}
    </div>
  );
}
