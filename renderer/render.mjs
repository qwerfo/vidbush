// Render one Vidbush MotionSpec with Remotion.
// Usage: node render.mjs spec.json out.mp4 [--alpha]
// --alpha renders ProRes 4444 with alpha, then converts to qtrle argb .mov (same as the Python renderer).
import { bundle } from "@remotion/bundler";
import { renderMedia, selectComposition } from "@remotion/renderer";
import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, renameSync, unlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

/** Chromium from VIDBUSH_CHROMIUM, else the Playwright cache at /opt/pw-browsers, else Remotion's own download. */
function findBrowser() {
  if (process.env.VIDBUSH_CHROMIUM) return process.env.VIDBUSH_CHROMIUM;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH || "/opt/pw-browsers";
  if (!existsSync(root)) return null;
  const candidates = [];
  for (const d of readdirSync(root)) {
    candidates.push(path.join(root, d, "chrome-headless-shell-linux64", "chrome-headless-shell"));
    candidates.push(path.join(root, d, "chrome-linux", "headless_shell"));
    candidates.push(path.join(root, d, "chrome-linux64", "chrome"));
    candidates.push(path.join(root, d, "chrome-linux", "chrome"));
  }
  return candidates.find((c) => existsSync(c)) ?? null;
}

const [specPath, outPath, ...flags] = process.argv.slice(2);
if (!specPath || !outPath) {
  console.error("usage: node render.mjs spec.json out.mp4 [--alpha]");
  process.exit(2);
}
const alpha = flags.includes("--alpha");
const spec = JSON.parse(readFileSync(specPath, "utf-8"));
const inputProps = { spec };
const browserExecutable = findBrowser();
const serveUrl = await bundle({ entryPoint: path.join(here, "src", "index.ts"), publicDir: path.join(here, "..", "assets") });
const composition = await selectComposition({ serveUrl, id: spec.kind.replace(/_/g, "-"), inputProps, browserExecutable });
const tmp = alpha ? outPath + ".prores.mov" : outPath;
await renderMedia({
  serveUrl, composition, inputProps, browserExecutable, outputLocation: tmp,
  codec: alpha ? "prores" : "h264",
  ...(alpha ? { proResProfile: "4444", imageFormat: "png", pixelFormat: "yuva444p10le" } : { crf: 18, pixelFormat: "yuv420p" }),
  chromiumOptions: { gl: "swangle" },
  logLevel: "error",
});
if (alpha) {
  const r = spawnSync("ffmpeg", ["-y", "-v", "error", "-i", tmp, "-c:v", "qtrle", "-pix_fmt", "argb", outPath + ".part.mov"]);
  unlinkSync(tmp);
  if (r.status !== 0) {
    console.error(String(r.stderr));
    process.exit(1);
  }
  renameSync(outPath + ".part.mov", outPath);
}
console.log(outPath);
