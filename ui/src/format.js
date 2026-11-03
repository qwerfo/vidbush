export const STAGES = [
  "intake", "research", "outline", "script", "narration", "shotlist", "sourcing",
  "judge", "visuals", "motion", "mix", "assembly", "packaging",
];

export const TYPE_COLORS = {
  footage: "#4f9dff",
  footage_overlay: "#38c8e0",
  image: "#56c271",
  image_overlay: "#2fb5a0",
  ai_image: "#a879ff",
  ai_video: "#ff6fb5",
  motion_graphic: "#f2b33d",
};

export const TYPE_LABELS = {
  footage: "Footage",
  footage_overlay: "Footage + overlay",
  image: "Image",
  image_overlay: "Image + overlay",
  ai_image: "AI image",
  ai_video: "AI video",
  motion_graphic: "Motion graphic",
};

export function clock(sec) {
  if (sec === null || sec === undefined || Number.isNaN(sec)) return "-";
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r < 10 ? "0" : ""}${r.toFixed(1)}`;
}

export function ago(ts) {
  if (!ts) return "";
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export function score(v) {
  return v === null || v === undefined ? "-" : Math.round(v);
}

export function opSummary(op) {
  const a = op.args || {};
  const parts = Object.entries(a).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
  return parts.join(", ");
}
