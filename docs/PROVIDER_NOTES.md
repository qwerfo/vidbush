# Provider notes

Quirks found on first contact with real providers (live-run-1, 2026-11-04, Windows 11 laptop).
Each line says what was seen and what Vidbush now does about it. Re-check anything here before
relying on it months later: relays change.

## CloseRouter (https://api.closerouter.dev/v1), primary text and vision

- Model ids are `provider/model` with dotted versions: `anthropic/claude-sonnet-5`,
  `anthropic/claude-opus-5`, `google/gemini-3.8-flash`. All three listed and answered.
- Anthropic replies usually carry no `usage.cost` (one long reply did, small JSON replies did not);
  Gemini replies always do. Vidbush now prices any reply without a cost from the provider's own
  `GET /models` price list (`pricing.unit: usd_per_million_tokens`), with cache reads and cache
  writes at their own rates (`vidbush/providers/llm.py` `price_usage`).
- Anthropic calls arrive with 4,000 to 11,000 hidden prompt tokens (mostly cached): the relay adds
  its own system prompt. It is cheap (cache read rate) but it can steer output: a plain-text
  request came back as `I'll write... <write_file><path>..</path><content>TEXT`. JSON calls with
  `response_format: json_object` came back clean. The JSON parser now unwraps a `<content>`
  scaffold before looking for the JSON object.
- No truncation without `max_tokens`: a 2,500 word request returned 3,545 tokens, `finish_reason:
  stop`, in 63 s. So Vidbush still sends no `max_tokens`; `options.max_tokens` exists for a relay
  that does truncate.
- Gemini 3.8 Flash wraps JSON in a ```json fence and spends 100 to 200 reasoning tokens even on
  tiny prompts. The fence is handled by the loose parser. Vision with a base64 JPEG works.

## Wellflow (https://api.wellflow.dev/v1), text and vision fallback

- Adam has two keys. The one named `WELLFLOW_CHEAP` answers HTTP 403 `model_not_in_channel
  (cheap)` for every model asked (gemini-3.8-flash, qwen3.8-max, deepseek-v4.1-flash,
  claude-sonnet-5). `WELLFLOW_API_KEY` serves all of them, so the fallback seats use it.
- Model ids are bare: `qwen3.8-max`, `gemini-3.8-flash`, `deepseek-v4.1-flash`. `GET /models`
  carries no prices, so Wellflow calls are costed with the generic per 1k token estimate.
- Slow: small requests take 30 to 100 s and sometimes pass a 30 s read timeout. RadioPlay measured
  that Wellflow drops a non-streamed connection near 240 s; streamed replies survive because the
  relay sends `: ping` keepalives. Both Wellflow providers set `options.stream: true`; Vidbush folds
  the SSE chunks back into one reply (`vidbush/providers/http.py` `post_stream`). The probe
  timeout is 120 s.

## Elevate (https://public-api.elevate.uno), primary images

- Not an OpenAI images API. `POST /v2/media {type: image, prompt, model, aspect_ratio:
  landscape|portrait|square, params}` returns `{success, data: {id, status}}`; poll
  `GET /v2/media/{id}?type=image` until `completed`, then download `data.result_url` (no auth).
  New provider type `elevate` in `vidbush/providers/image.py`.
- Errors can arrive as HTTP 200 with `{success: false, error: {code, message}}`. `RATE_LIMIT*`
  is a 429 (back off), `INVALID_REQUEST` is a refusal, "please retry" messages are transient.
- Cloudflare in front of Elevate answers error 1010 to the python-requests default User-Agent.
  Every provider request now sends a browser-like User-Agent (`vidbush/providers/http.py`).
- Model id for GPT Image 2.5 is `gpt-image-2.5-fast` (4 credits, about 30 to 50 s) or
  `gpt-image-2.5-thinking` (7 credits). Subscription: 3 at once, 200 an hour for this key; cost
  per call is recorded as 0.
- The picture comes back 1376x768 with a small sparkle watermark in the bottom right corner.
  `options.edge_trim: 0.05` crops 5 percent from each edge before the 16:9 fit.

## Algrow (https://api.algrow.online), fallback images

- `POST /api/generate-image {prompt, model: gpt-image-2.5, aspect_ratio: "16:9", fast: false}`
  returns `{success, job_id, credits_used: 0.35}`; poll `GET /api/job-status/{job_id}` until
  `completed`; the image is the first of `image_urls` (hosted, no auth). New provider type
  `algrow`. 0.35 credits, 0.01085 USD. Jobs normally take 35 to 60 s but RadioPlay saw up to
  about 7 minutes, so the timeout is 600 s.

## ElevenLabs (direct key), narration

- `GET /v1/models` lists `eleven_v4` (newest), `eleven_v4_turbo`, `eleven_v3`,
  `eleven_multilingual_v2` and older. `eleven_v4` works on `/with-timestamps` (alignment and
  normalized_alignment returned) and accepts `previous_text`.
- `options.model_preference: [eleven_v4, eleven_multilingual_v2]`. If the API refuses the auto
  picked model, the provider switches to multilingual v2 for the rest of the run (never per
  chunk, so one video never mixes models). Quota and auth errors are not treated as model errors.

## Windows host

- Pillow wheels on Windows have no libraqm, so `features=["tnum"]` raised KeyError and every
  number card, bar chart, line chart and recap grid fell back to a plain text overlay. Font
  features are now dropped when raqm is missing (digits use proportional widths).
- numpy 2.5 needs Python 3.12; `pyproject.toml` pins numpy 2.4.6 on Python 3.11.
- Paths rebased onto a Docker root (`/app/output/...`) kept Windows separators; rebasing now keeps
  the target root's separator style.
