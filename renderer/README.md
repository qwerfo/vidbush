# Vidbush Remotion renderer (optional)

Vidbush renders motion graphics with its built-in Python renderer
(`vidbush/render/mg_python.py`) by default. This folder is an optional
alternative that renders the same `MotionSpec` JSON with
[Remotion](https://www.remotion.dev) (React + headless Chromium). The JSON
spec is the contract: any renderer that turns a MotionSpec into a clip of
exactly `duration` seconds at `width x height` and `fps` (with alpha when
asked) can replace the built-in one.

## Enable it

1. Install Node 18 or newer, then in this folder run `npm install`
   (`node_modules/` is git ignored).
2. A Chromium is needed. `render.mjs` looks for, in order: the
   `VIDBUSH_CHROMIUM` environment variable (path to a chrome or
   chrome-headless-shell binary), a Playwright browser cache under
   `PLAYWRIGHT_BROWSERS_PATH` or `/opt/pw-browsers`, and otherwise lets
   Remotion download its own headless shell.
3. Set `render.renderer: remotion` in `config/vidbush.yaml`.

If a Remotion render fails for any reason (Node missing, no browser, a
React error) the motion stage logs a warning and renders that spec with the
Python renderer instead, so a production never stops on it.

## Use it directly

```
node render.mjs spec.json out.mp4           # H.264 yuv420p
node render.mjs spec.json out.mov --alpha   # qtrle argb with transparency
```

## Layout

- `src/Root.tsx` registers one composition per MotionKind (ids use dashes,
  for example `bar-chart`, because Remotion ids cannot contain underscores).
  Size, fps and length come from the spec passed as input props.
- `src/compositions/*.tsx` hold the compositions; `src/theme.ts` mirrors the
  three palettes in `vidbush/mg/palettes.py`.
- Fonts (Inter) and Natural Earth data are served from the repository
  `assets/` folder (bundled as the public dir).

## Notes

Each call bundles the project (about 8 to 12 seconds of overhead), so the
Python renderer is faster for long videos. The Python renderer is also the
reference design: maps there have terrain texture, coast halos, lakes,
rivers, glows and label collision avoidance that this project keeps simpler.
