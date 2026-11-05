import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { useJobWatcher } from "../hooks.js";
import { TYPE_COLORS, TYPE_LABELS, clock, opSummary, score } from "../format.js";

// Review sheet: final.mp4 on top, one row per shot. Every change saves itself to review.json.
// Keys: left and right move between shots, 1 to 5 set the verdict, c edits the comment,
// space plays or pauses, Esc leaves a text box. Comment boxes have no length limit.

const VERDICT_KEYS = ["good", "wrong_picture", "wrong_timing", "quality_problem", "other"];
const COMMENT_DELAY = 600;

function isTyping(el) {
  return el && (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable);
}

// A textarea that grows with its text, so a long comment is never hidden behind a scrollbar.
function Grow({ value, onChange, onBlur, inputRef, ...rest }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [value]);
  const setRef = useCallback((el) => {
    ref.current = el;
    if (inputRef) inputRef(el);
  }, [inputRef]);
  return <textarea ref={setRef} rows={1} value={value} onChange={onChange} onBlur={onBlur} {...rest} />;
}

function hostOf(url) {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : "";
  } catch {
    return "";
  }
}

// Saves run one after another per key, so a slow request never overwrites a newer value.
function useSaver(onSaved, onError) {
  const chains = useRef({});
  const [pending, setPending] = useState(0);
  const run = useCallback((key, fn) => {
    setPending((n) => n + 1);
    const prev = chains.current[key] || Promise.resolve();
    const next = prev.then(fn).then(
      (res) => { onSaved(res); },
      (e) => { onError(e); },
    ).finally(() => setPending((n) => n - 1));
    chains.current[key] = next;
    return next;
  }, [onSaved, onError]);
  return [run, pending];
}

function ShotRow({ shot, entry, index, current, playing, threshold, verdicts, rejects, onFocus, onSeek,
  onVerdict, onReject, onComment, onCommentBlur, commentRef }) {
  const sel = (shot.candidates || []).find((c) => c.selected);
  const rejected = (shot.candidates || []).filter((c) => c.reject_reason && !c.selected);
  const [showAll, setShowAll] = useState(false);
  const low = shot.judge_score !== null && shot.judge_score !== undefined && shot.judge_score < threshold;
  const verdict = entry.verdict || "";
  const cls = ["review-row", current && "current", playing && "playing", verdict && `v-${verdict}`].filter(Boolean).join(" ");
  const shown = showAll ? rejected : rejected.slice(0, 3);
  return (
    <article className={cls} data-shot={shot.id} aria-current={current ? "true" : undefined}
      onMouseDown={() => onFocus(index)} onFocus={() => onFocus(index)}>
      <button className="review-thumb" onClick={() => onSeek(shot)} title="Play from this shot">
        <img src={shot.thumb_url} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
        <span className="review-num">{shot.index + 1}</span>
      </button>
      <div className="review-meta">
        <div className="review-line">
          <button className="time-link" onClick={() => onSeek(shot)} title="Seek the player here">{clock(shot.start)}</button>
          <span className="tag" style={{ background: TYPE_COLORS[shot.type] }}>{TYPE_LABELS[shot.type] || shot.type}</span>
          <span className="muted small">{shot.duration.toFixed(1)} s</span>
          {shot.overlay && <span className="chip">overlay: {shot.overlay}</span>}
          {shot.motion && <span className="chip">graphic: {shot.motion}</span>}
        </div>
        <p className="review-intent">{shot.intent}</p>
        <dl className="facts review-facts">
          <dt>Source</dt>
          <dd>
            {sel ? (
              <>
                {sel.url ? <a href={sel.url} target="_blank" rel="noreferrer">{hostOf(sel.url) || sel.source}</a> : sel.source}
                {", "}<span title={sel.licence}>{sel.licence || "licence unknown"}</span>
                {sel.title && <span className="muted">, {sel.title}</span>}
              </>
            ) : <span className="muted">none selected</span>}
          </dd>
          <dt>Judge</dt>
          <dd>
            <span className={`score-badge ${low ? "low" : ""}`}>{score(shot.judge_score)}</span>
            {sel && sel.judge_reason && <span className="muted small"> {sel.judge_reason}</span>}
            {(shot.flags || []).map((f) => <span key={f} className="chip warn-chip">{f}</span>)}
          </dd>
          {rejected.length > 0 && (
            <>
              <dt>Rejected</dt>
              <dd>
                <ul className="reject-list">
                  {shown.map((c) => (
                    <li key={c.id}>
                      <span className="muted">{hostOf(c.url) || c.source}:</span> <span className="reject-text">{c.reject_reason}</span>
                    </li>
                  ))}
                </ul>
                {rejected.length > 3 && (
                  <button className="link-btn" onClick={() => setShowAll(!showAll)}>
                    {showAll ? "Show fewer" : `${rejected.length - 3} more`}
                  </button>
                )}
              </dd>
            </>
          )}
        </dl>
      </div>
      <div className="review-controls">
        <div className="verdicts" role="radiogroup" aria-label={`Verdict for shot ${shot.index + 1}`}>
          {verdicts.map((v, i) => (
            <button key={v.key} role="radio" aria-checked={verdict === v.key}
              className={`verdict-btn vb-${v.key} ${verdict === v.key ? "on" : ""}`}
              onClick={() => onVerdict(shot, verdict === v.key ? null : v.key)} title={`Key ${i + 1}`}>
              <kbd>{i + 1}</kbd> {v.label}
            </button>
          ))}
        </div>
        <div className="reject-checks" aria-label="Hard rejects">
          {rejects.map((r) => (
            <label key={r.key} className="check">
              <input type="checkbox" checked={(entry.rejects || []).includes(r.key)}
                onChange={(e) => onReject(shot, r.key, e.target.checked)} />
              {r.label}
            </label>
          ))}
        </div>
        <Grow className="review-comment" value={entry.comment || ""} inputRef={commentRef}
          placeholder="Comment (no limit): what is wrong, what you want instead"
          onChange={(e) => onComment(shot, e.target.value)} onBlur={() => onCommentBlur(shot)} />
        {entry.fix && <div className="small muted">Fix queued: {entry.fix.op}{entry.fix.job_id ? ` (job ${entry.fix.job_id})` : ""}</div>}
      </div>
    </article>
  );
}

function FixesPanel({ plan, busy, projectId, onApply, onClose }) {
  const { ops, timing, comments, skipped } = plan;
  return (
    <section className="card fixes" aria-label="Planned fixes">
      <div className="card-head">
        <h3>Apply fixes</h3>
        <button className="btn ghost" onClick={onClose}>Close</button>
      </div>
      {ops.length === 0 && <p className="muted">No verdict maps to a safe edit yet. Wrong picture, or quality problem with a hard reject ticked, become edits.</p>}
      {ops.length > 0 && (
        <>
          <p className="small muted">These edits run as one job. Only the changed segments render again.</p>
          <ol className="ops">
            {ops.map((o, i) => (
              <li key={i}>
                <code>{o.op}</code> <span className="small">{opSummary(o)}</span>
                {o.reason && <div className="small muted">{o.reason}</div>}
              </li>
            ))}
          </ol>
        </>
      )}
      {timing.length > 0 && (
        <>
          <h4>Wrong timing ({timing.length})</h4>
          <ul className="plain-list">
            {timing.map((t) => (
              <li key={t.shot_id}>
                Shot {t.number} at {clock(t.start)}, now {t.duration.toFixed(2)} s
                {t.comment && <span className="muted small"> {t.comment}</span>}
                {" "}<a className="btn small-btn" href={`#/p/${encodeURIComponent(projectId)}/${encodeURIComponent(t.shot_id)}`}>Open duration editor</a>
              </li>
            ))}
          </ul>
        </>
      )}
      {comments.length > 0 && (
        <details className="advanced">
          <summary>{comments.length} verdict(s) stay a comment</summary>
          <ul className="plain-list">
            {comments.map((c) => <li key={c.shot_id} className="small">Shot {c.number}: {c.verdict_label} ({c.why})</li>)}
          </ul>
        </details>
      )}
      {skipped.length > 0 && <p className="small muted">{skipped.length} shot(s) already have a fix queued since their last change.</p>}
      <div className="actions">
        <button className="btn ghost" onClick={onClose}>Discard</button>
        <button className="btn primary" disabled={busy || ops.length === 0} onClick={() => onApply(ops)}>Apply {ops.length} edit(s)</button>
      </div>
    </section>
  );
}

export default function Review({ id }) {
  const [project, setProject] = useState(null);
  const [shots, setShots] = useState(null);
  const [review, setReview] = useState(null);
  const [error, setError] = useState(null);
  const [cur, setCur] = useState(0);
  const [playingId, setPlayingId] = useState(null);
  const [saveState, setSaveState] = useState({ kind: "idle", at: null, text: "" });
  const [plan, setPlan] = useState(null);
  const [notice, setNotice] = useState(null);
  const [docked, setDocked] = useState(false);
  const player = useRef(null);
  const playerSlot = useRef(null);
  const list = useRef(null);
  const commentRefs = useRef({});
  const timers = useRef({});
  const latest = useRef({});
  const setters = useRef({});
  const commentSetter = (shotId) => {
    if (!setters.current[shotId]) setters.current[shotId] = (el) => { commentRefs.current[shotId] = el; };
    return setters.current[shotId];
  };

  const load = useCallback(async () => {
    try {
      const [p, s, r] = await Promise.all([api.project(id), api.shots(id), api.review(id)]);
      setProject(p);
      setShots(s.shots);
      setReview(r);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [id]);
  useEffect(() => { load(); }, [load]);

  const onSaved = useCallback((res) => {
    if (res && res.shot) {
      setReview((r) => r && ({ ...r, reviewed: res.reviewed, total: res.total,
        shots: r.shots.map((e) => (e.shot_id === res.shot.shot_id ? { ...e, fix: res.shot.fix, snapshot: res.shot.snapshot, updated_at: res.shot.updated_at } : e)) }));
    }
    setSaveState({ kind: "saved", at: new Date(), text: "" });
  }, []);
  const onSaveError = useCallback((e) => setSaveState({ kind: "error", at: new Date(), text: e.message }), []);
  const [save, pending] = useSaver(onSaved, onSaveError);
  useEffect(() => { if (pending > 0) setSaveState((s) => ({ ...s, kind: "saving" })); }, [pending]);

  const watcher = useJobWatcher(id, (job) => {
    setNotice(job.status === "failed"
      ? { kind: "error", text: `Fixes failed: ${(job.error || "").split("\n")[0]}` }
      : { kind: "ok", text: "Fixes applied and re-rendered. Review the changed shots again." });
    load();
  });

  const entries = useMemo(() => {
    const m = {};
    (review ? review.shots : []).forEach((e) => { m[e.shot_id] = e; });
    return m;
  }, [review]);
  const rows = shots || [];
  const threshold = review ? review.threshold : 70;
  const verdicts = review ? review.verdicts : [];
  const rejects = review ? review.rejects : [];

  const patchLocal = useCallback((shotId, patch) => {
    setReview((r) => {
      if (!r) return r;
      const shotsNext = r.shots.map((e) => (e.shot_id === shotId ? { ...e, ...patch } : e));
      return { ...r, shots: shotsNext, reviewed: shotsNext.filter((e) => e.in_shotlist && e.verdict).length };
    });
  }, []);

  const setVerdict = useCallback((shot, verdict) => {
    patchLocal(shot.id, { verdict });
    save(shot.id, () => api.saveShotReview(id, shot.id, verdict ? { verdict } : { clear_verdict: true }));
  }, [id, patchLocal, save]);

  const setReject = useCallback((shot, key, on) => {
    const now = (entries[shot.id] && entries[shot.id].rejects) || [];
    const next = on ? [...new Set([...now, key])] : now.filter((r) => r !== key);
    patchLocal(shot.id, { rejects: next });
    save(shot.id, () => api.saveShotReview(id, shot.id, { rejects: next }));
  }, [entries, id, patchLocal, save]);

  const flushComment = useCallback((shotId) => {
    clearTimeout(timers.current[shotId]);
    delete timers.current[shotId];
    if (!(shotId in latest.current)) return;
    const text = latest.current[shotId];
    delete latest.current[shotId];
    save(shotId, () => api.saveShotReview(id, shotId, { comment: text }));
  }, [id, save]);

  const setComment = useCallback((shot, text) => {
    patchLocal(shot.id, { comment: text });
    latest.current[shot.id] = text;
    setSaveState((s) => ({ ...s, kind: "editing" }));
    clearTimeout(timers.current[shot.id]);
    timers.current[shot.id] = setTimeout(() => flushComment(shot.id), COMMENT_DELAY);
  }, [flushComment, patchLocal]);

  // project comment: same debounce under its own key
  const setProjectComment = useCallback((text) => {
    setReview((r) => r && ({ ...r, project_comment: text }));
    latest.current.__project = text;
    clearTimeout(timers.current.__project);
    timers.current.__project = setTimeout(() => {
      const t = latest.current.__project;
      delete latest.current.__project;
      save("__project", () => api.saveReviewComment(id, t));
    }, COMMENT_DELAY);
  }, [id, save]);
  const flushProject = useCallback(() => {
    if (!("__project" in latest.current)) return;
    clearTimeout(timers.current.__project);
    const t = latest.current.__project;
    delete latest.current.__project;
    save("__project", () => api.saveReviewComment(id, t));
  }, [id, save]);

  // never lose typed text: flush on leave
  useEffect(() => {
    const flushAll = () => {
      Object.values(timers.current).forEach(clearTimeout);
      timers.current = {};
      Object.keys(latest.current).forEach((k) => {
        const body = { comment: latest.current[k] };
        const url = k === "__project" ? `/api/projects/${encodeURIComponent(id)}/review/comment`
          : `/api/projects/${encodeURIComponent(id)}/review/shots/${encodeURIComponent(k)}`;
        fetch(url, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), keepalive: true })
          .catch(() => {});
      });
      latest.current = {};
    };
    window.addEventListener("beforeunload", flushAll);
    return () => { window.removeEventListener("beforeunload", flushAll); flushAll(); };
  }, [id]);

  const seek = useCallback((shot) => {
    const v = player.current;
    if (!v) return;
    v.currentTime = shot.start + 0.01;
  }, []);

  const move = useCallback((i) => {
    if (!rows.length) return;
    const n = Math.min(rows.length - 1, Math.max(0, i));
    setCur(n);
    seek(rows[n]);
    const el = list.current && list.current.querySelector(`[data-shot="${rows[n].id}"]`);
    if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [rows, seek]);

  useEffect(() => {
    function onKey(e) {
      if (e.altKey || e.metaKey || e.ctrlKey) return;
      if (isTyping(e.target)) {
        if (e.key === "Escape") e.target.blur();
        return;
      }
      if (e.target.tagName === "VIDEO" && (e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === " ")) return;
      const shot = rows[cur];
      if (e.key === "ArrowRight" || e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); move(cur + 1); return; }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); move(cur - 1); return; }
      if (!shot) return;
      const n = Number(e.key);
      if (n >= 1 && n <= VERDICT_KEYS.length) {
        e.preventDefault();
        const v = VERDICT_KEYS[n - 1];
        const now = entries[shot.id] && entries[shot.id].verdict;
        setVerdict(shot, now === v ? null : v);
        return;
      }
      if (e.key === "c") {
        const el = commentRefs.current[shot.id];
        if (el) { e.preventDefault(); el.focus(); }
        return;
      }
      if (e.key === " " && player.current) {
        e.preventDefault();
        if (player.current.paused) player.current.play().catch(() => {}); else player.current.pause();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows, cur, entries, move, setVerdict]);

  // dock the player in a corner once its slot scrolls out of view
  useEffect(() => {
    const el = playerSlot.current;
    if (!el || typeof IntersectionObserver === "undefined") return undefined;
    const io = new IntersectionObserver(([en]) => setDocked(!en.isIntersecting), { threshold: 0, rootMargin: "-120px 0px 0px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [project && project.final_url]);

  function onTime() {
    const t = player.current ? player.current.currentTime : 0;
    const s = rows.find((x) => t >= x.start && t < x.end);
    if (s && s.id !== playingId) setPlayingId(s.id);
  }

  async function openFixes() {
    setNotice(null);
    Object.keys(latest.current).forEach((k) => (k === "__project" ? flushProject() : flushComment(k)));
    try {
      setPlan(await api.reviewFixes(id));
    } catch (e) {
      setNotice({ kind: "error", text: e.message });
    }
  }

  async function applyFixes(ops) {
    try {
      const res = await api.applyReviewFixes(id, ops);
      setPlan(null);
      setNotice(null);
      watcher.watch(res.job_id);
      load();
    } catch (e) {
      setNotice({ kind: "error", text: e.message });
    }
  }

  async function exportMd() {
    Object.keys(latest.current).forEach((k) => (k === "__project" ? flushProject() : flushComment(k)));
    try {
      const res = await api.exportReview(id);
      setNotice({ kind: "ok", text: "Wrote review.md.", url: res.url });
    } catch (e) {
      setNotice({ kind: "error", text: e.message });
    }
  }

  if (error && !review) return <div className="page"><div className="error">{error}</div></div>;
  if (!review || !shots || !project) return <div className="page muted">Loading</div>;

  const pct = review.total ? Math.round((100 * review.reviewed) / review.total) : 0;
  const saveText = { idle: "All changes save automatically", editing: "Typing", saving: "Saving",
    saved: `Saved ${saveState.at ? saveState.at.toLocaleTimeString() : ""}`, error: `Save failed: ${saveState.text}` }[saveState.kind];
  return (
    <div className="page review-page">
      <div className="review-head" role="region" aria-label="Review progress">
        <div className="review-head-row">
          <div className="review-title">
            <a href={`#/p/${encodeURIComponent(id)}`} className="muted small">Back to project</a>
            <h1>Review: {project.title || id}</h1>
          </div>
          <div className="review-progress" aria-live="polite">
            <span><strong>{review.reviewed}</strong> of {review.total} shots reviewed</span>
            <div className="mini-progress wide"><span style={{ width: `${pct}%` }} /></div>
            <span className={`small ${saveState.kind === "error" ? "bad" : "muted"}`}>{saveText}</span>
          </div>
          <div className="head-actions">
            <button className="btn" onClick={exportMd}>Export review.md</button>
            <button className="btn primary" disabled={watcher.watching} onClick={openFixes}>Apply fixes</button>
          </div>
        </div>
        <Grow className="project-comment" value={review.project_comment || ""}
          placeholder="Project comment (no limit): overall notes on this render"
          onChange={(e) => setProjectComment(e.target.value)} onBlur={flushProject} />
        <div className="small muted keys-hint">
          <kbd>Left</kbd> <kbd>Right</kbd> move between shots, <kbd>1</kbd> to <kbd>5</kbd> set the verdict,
          {" "}<kbd>c</kbd> comment, <kbd>Space</kbd> play or pause, <kbd>Esc</kbd> leave a text box.
        </div>
      </div>

      {watcher.watching && (
        <div className="banner working" role="status">
          <span className="spinner" aria-hidden="true" /> Applying fixes. Only the changed segments render again.
          {watcher.job && <span className="muted small"> Job {watcher.job.id}: {watcher.job.status}</span>}
        </div>
      )}
      {notice && (
        <div className={notice.kind === "error" ? "error" : "banner ok"}>
          {notice.text} {notice.url && <a href={notice.url} target="_blank" rel="noreferrer">Open review.md</a>}
        </div>
      )}
      {plan && <FixesPanel plan={plan} busy={watcher.watching} projectId={id} onApply={applyFixes} onClose={() => setPlan(null)} />}

      <div className="review-player-slot" ref={playerSlot}>
        {project.final_url ? (
          <div className={`review-player ${docked ? "docked" : ""}`}>
            <video ref={player} className="player" src={project.final_url} controls preload="metadata" onTimeUpdate={onTime} />
          </div>
        ) : (
          <div className="player placeholder">No final video yet. Render the project first.</div>
        )}
      </div>

      <div className="review-list" ref={list}>
        {rows.map((s, i) => (
          <ShotRow key={s.id} shot={s} index={i} entry={entries[s.id] || {}} current={i === cur} playing={s.id === playingId}
            threshold={threshold} verdicts={verdicts} rejects={rejects} onFocus={setCur} onSeek={(shot) => { setCur(i); seek(shot); }}
            onVerdict={setVerdict} onReject={setReject} onComment={setComment} onCommentBlur={(shot) => flushComment(shot.id)}
            commentRef={commentSetter(s.id)} />
        ))}
      </div>
    </div>
  );
}
