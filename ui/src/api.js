// Thin JSON client for the Vidbush API. Every call throws an Error with the server's detail text.

async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    throw new Error(detailText(data) || `${res.status} ${res.statusText}`);
  }
  return data;
}

function detailText(data) {
  if (!data) return "";
  const d = typeof data === "object" ? data.detail : data;
  if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
  return typeof d === "string" ? d : JSON.stringify(d);
}

const enc = encodeURIComponent;

export const api = {
  health: () => request("GET", "/api/health"),
  settings: () => request("GET", "/api/settings"),
  projects: () => request("GET", "/api/projects"),
  createProject: (form) => request("POST", "/api/projects", form),
  project: (id) => request("GET", `/api/projects/${enc(id)}`),
  log: (id, n = 200) => request("GET", `/api/projects/${enc(id)}/log?n=${n}`),
  jobs: (id) => request("GET", `/api/projects/${enc(id)}/jobs`),
  job: (id, jobId) => request("GET", `/api/projects/${enc(id)}/jobs/${jobId}`),
  resume: (id) => request("POST", `/api/projects/${enc(id)}/resume`),
  shots: (id) => request("GET", `/api/projects/${enc(id)}/shots`),
  swap: (id, shot, candidate_id) =>
    request("POST", `/api/projects/${enc(id)}/shots/${enc(shot)}/swap`, { candidate_id }),
  upload: (id, shot, file) => {
    const f = new FormData();
    f.append("file", file);
    return request("POST", `/api/projects/${enc(id)}/shots/${enc(shot)}/upload`, f);
  },
  regenerate: (id, shot, prompt_delta) =>
    request("POST", `/api/projects/${enc(id)}/shots/${enc(shot)}/regenerate`, { prompt_delta }),
  duration: (id, shot, seconds) =>
    request("POST", `/api/projects/${enc(id)}/shots/${enc(shot)}/duration`, { seconds }),
  removeOverlay: (id, shot) => request("POST", `/api/projects/${enc(id)}/shots/${enc(shot)}/remove_overlay`),
  ops: (id, ops) => request("POST", `/api/projects/${enc(id)}/ops`, { ops }),
  chat: (id, message, apply = false) => request("POST", `/api/projects/${enc(id)}/chat`, { message, apply }),
  brands: () => request("GET", "/api/brands"),
  brand: (id) => request("GET", `/api/brands/${enc(id)}`),
  saveBrand: (id, data) => request("PUT", `/api/brands/${enc(id)}`, data),
  styleProfiles: () => request("GET", "/api/style_profiles"),
};
