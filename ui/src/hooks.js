import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

// Current location hash split into parts: "#/p/abc" -> ["p", "abc"].
export function useRoute() {
  const read = () => (window.location.hash.replace(/^#\/?/, "") || "projects").split("/").map(decodeURIComponent);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function go(path) {
  window.location.hash = `#/${path}`;
}

// Load data with fn(); re-run every `interval` ms while interval is truthy. Returns [data, error, reload].
export function usePoll(fn, deps, interval) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const load = useCallback(async () => {
    try {
      setData(await fnRef.current());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);
  useEffect(() => {
    setData(null);
    load();
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!interval) return undefined;
    const t = setInterval(load, interval);
    return () => clearInterval(t);
  }, [interval, load]);
  return [data, error, load];
}

// Poll one job until it is done or failed, then call onDone(job).
export function useJobWatcher(projectId, onDone) {
  const [job, setJob] = useState(null);
  const [jobId, setJobId] = useState(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;
  useEffect(() => {
    if (!jobId) return undefined;
    let stop = false;
    const tick = async () => {
      try {
        const j = await api.job(projectId, jobId);
        if (stop) return;
        setJob(j);
        if (j.status === "done" || j.status === "failed") {
          setJobId(null);
          doneRef.current && doneRef.current(j);
          return;
        }
      } catch {
        // transient network error: keep polling
      }
      if (!stop) setTimeout(tick, 2000);
    };
    tick();
    return () => {
      stop = true;
    };
  }, [jobId, projectId]);
  return { job, watching: Boolean(jobId), watch: setJobId, clear: () => setJob(null) };
}
