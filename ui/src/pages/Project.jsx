import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useJobWatcher, usePoll } from "../hooks.js";
import { opSummary } from "../format.js";
import StatusPill from "../components/StatusPill.jsx";
import StageBar from "../components/StageBar.jsx";
import LogTail from "../components/LogTail.jsx";
import ShotStrip from "../components/ShotStrip.jsx";
import ShotPanel from "../components/ShotPanel.jsx";
import ChatPanel from "../components/ChatPanel.jsx";
import Outputs from "../components/Outputs.jsx";

let thresholdCache = null;

function useThreshold() {
  const [t, setT] = useState(thresholdCache || 70);
  useEffect(() => {
    if (thresholdCache) return;
    api.settings().then((s) => { thresholdCache = s.judge.threshold; setT(thresholdCache); }).catch(() => {});
  }, []);
  return t;
}

function isTyping(el) {
  return el && (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable);
}

export default function Project({ id, initialShot }) {
  const [live, setLive] = useState(true);
  const [project, error, reloadProject] = usePoll(() => api.project(id), [id], live ? 2000 : 0);
  const [shots, setShots] = useState(null);
  const [shotId, setShotId] = useState(null);
  const [playingId, setPlayingId] = useState(null);
  const [tab, setTab] = useState("shot");
  const [jobLabel, setJobLabel] = useState("");
  const [notice, setNotice] = useState(null);
  const player = useRef(null);
  const resumeAt = useRef(null);
  const threshold = useThreshold();
  const [focusDuration, setFocusDuration] = useState(false);

  const loadShots = useCallback(() => api.shots(id).then(setShots).catch(() => setShots(null)), [id]);

  const watcher = useJobWatcher(id, (job) => {
    if (player.current) resumeAt.current = player.current.currentTime;
    setNotice(job.status === "failed"
      ? { kind: "error", text: `Re-render failed: ${(job.error || "").split("\n")[0]}` }
      : { kind: "ok", text: "Re-render finished. The player shows the new cut." });
    reloadProject();
    loadShots();
  });

  const running = project && (project.status === "running" || Boolean(project.active_job));
  useEffect(() => { setLive(Boolean(!project || running)); }, [project, running]);
  const status = project && project.status;
  useEffect(() => { if (project) loadShots(); }, [status, project && project.final_url, loadShots]); // eslint-disable-line react-hooks/exhaustive-deps

  // pick up an edit job that was already running when the page opened
  useEffect(() => {
    const j = project && project.active_job;
    if (j && j.kind === "apply_ops" && !watcher.watching) {
      setJobLabel("Applying edits");
      watcher.watch(j.id);
    }
  }, [project && project.active_job && project.active_job.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const list = shots ? shots.shots : [];
  const shot = list.find((s) => s.id === shotId) || null;

  // #/p/<id>/<shot_id> (the review page's "Open duration editor") opens that shot's panel
  useEffect(() => {
    if (!initialShot || !list.length) return;
    const s = list.find((x) => x.id === initialShot);
    if (s) {
      setShotId(s.id);
      setTab("shot");
      setFocusDuration(true);
      if (player.current) player.current.currentTime = s.start + 0.01;
    }
  }, [initialShot, list.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const pick = useCallback((s) => {
    setShotId(s.id);
    setTab("shot");
    if (player.current) player.current.currentTime = s.start + 0.01;
  }, []);

  useEffect(() => {
    function onKey(e) {
      if (isTyping(e.target) || e.altKey || e.metaKey || e.ctrlKey) return;
      if (e.key === "Escape") setShotId(null);
      if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && list.length && e.target.tagName !== "VIDEO") {
        const i = Math.max(0, list.findIndex((s) => s.id === shotId));
        const next = list[Math.min(list.length - 1, Math.max(0, i + (e.key === "ArrowRight" ? 1 : -1)))];
        if (next) { e.preventDefault(); pick(next); }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [list, shotId, pick]);

  function onTime() {
    const t = player.current ? player.current.currentTime : 0;
    const s = list.find((x) => t >= x.start && t < x.end);
    if (s && s.id !== playingId) setPlayingId(s.id);
  }

  function onJob(jobId, label) {
    setNotice(null);
    setJobLabel(label);
    watcher.watch(jobId);
    reloadProject();
  }

  async function resume() {
    try {
      await api.resume(id);
      setLive(true);
      reloadProject();
    } catch (e) {
      setNotice({ kind: "error", text: e.message });
    }
  }

  if (error && !project) return <div className="page"><div className="error">{error}</div></div>;
  if (!project) return <div className="page muted">Loading</div>;

  const busy = watcher.watching;
  const failedStage = Object.values(project.stages || {}).find((r) => r.status === "failed");
  const canResume = !project.active_job && (project.status === "failed" || project.status === "created");
  const lastEdit = project.last_edit;
  return (
    <div className="page project">
      <div className="page-head">
        <div>
          <a href="#/projects" className="muted small">Projects</a>
          <h1>{project.title || project.id}</h1>
          <div className="muted small">{project.id}, {project.input_mode}, {project.format === "top10" ? "Top 10" : "documentary"}, brand {project.brand_id}, target {project.target_minutes} min</div>
        </div>
        <div className="head-actions">
          {project.review && (
            <span className="small muted review-count">Reviewed: {project.review.reviewed} of {project.review.total}</span>
          )}
          {project.final_url && (
            <a className="btn" href={`#/review/${encodeURIComponent(id)}`}>{project.review ? "Continue review" : "Review"}</a>
          )}
          <StatusPill status={busy ? "running" : project.status} label={busy ? "re-rendering" : undefined} />
          {canResume && <button className="btn primary" onClick={resume}>Resume</button>}
        </div>
      </div>
      <StageBar project={project} />
      {project.status === "needs_review" && (
        <div className="banner warnbox" role="status">Finished, but some seats fell back to mock output while real providers are configured. Check the QA report before publishing.</div>
      )}
      {failedStage && project.status === "failed" && (
        <div className="error">Stage {failedStage.name} failed: {failedStage.error}</div>
      )}
      {busy && (
        <div className="banner working" role="status">
          <span className="spinner" aria-hidden="true" /> {jobLabel || "Re-rendering"}. Only the changed segments render again.
          {watcher.job && <span className="muted small"> Job {watcher.job.id}: {watcher.job.status}</span>}
        </div>
      )}
      {notice && <div className={notice.kind === "error" ? "error" : "banner ok"}>{notice.text}</div>}
      {lastEdit && lastEdit.result && (lastEdit.result.errors || []).length > 0 && !busy && (
        <div className="banner warnbox">Last edit reported: {lastEdit.result.errors.join("; ")}</div>
      )}

      <div className={`workspace ${shot || tab === "chat" ? "with-panel" : ""}`}>
        <section className="main-col">
          {project.final_url ? (
            <video ref={player} className="player" src={project.final_url} controls preload="metadata"
              onTimeUpdate={onTime}
              onLoadedMetadata={() => {
                if (resumeAt.current !== null && player.current) {
                  player.current.currentTime = resumeAt.current;
                  resumeAt.current = null;
                }
              }} />
          ) : (
            <div className="player placeholder">
              {running ? "Rendering. The video appears here when assembly finishes." : "No final video yet."}
            </div>
          )}
          {list.length > 0 && (
            <>
              <div className="strip-head">
                <span>{list.length} shots, {shots.duration.toFixed(1)} s</span>
                <span className="muted small">Click a shot or use the arrow keys. Esc closes the panel.</span>
                <span className="grow" />
                <button className={`btn ${tab === "chat" ? "primary" : ""}`} onClick={() => setTab(tab === "chat" ? "shot" : "chat")}>Chat revisions</button>
              </div>
              <ShotStrip shots={list} selectedId={shotId} playingId={playingId} threshold={threshold} onPick={pick} />
            </>
          )}
          <Outputs outputs={project.outputs} finalUrl={project.final_url} />
          {lastEdit && (
            <details className="card">
              <summary>Last edit ({(lastEdit.ops || []).length} op(s), {lastEdit.ok ? "ok" : "with problems"})</summary>
              <ol className="ops">
                {(lastEdit.ops || []).map((o, i) => <li key={i}><code>{o.op}</code> <span className="small">{opSummary(o)}</span></li>)}
              </ol>
              {lastEdit.error && <div className="error">{lastEdit.error}</div>}
            </details>
          )}
          <details className="card" open={running || project.status === "failed"}>
            <summary>Log</summary>
            <LogTail id={id} live={running || busy} />
          </details>
        </section>
        {tab === "chat" && <ChatPanel projectId={id} busy={busy} onJob={onJob} />}
        {tab !== "chat" && shot && (
          <ShotPanel projectId={id} shot={shot} busy={busy} onJob={onJob} focusDuration={focusDuration && shot.id === initialShot}
            onClose={() => { setShotId(null); setFocusDuration(false); }} />
        )}
      </div>
    </div>
  );
}
