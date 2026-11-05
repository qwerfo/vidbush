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

## Ported from RadioPlay (2026-11-05, parallel1 lane), ported, unverified live

Read from RadioPlay's working clients (`gateway/adapters/algrow_tts.py`, `anthropic_chat.py`, `_http.py`,
`config/roles.yaml`, `config/audio.yaml`, `docs/PROVIDER_FACTS.md`). None of this was sent live from Vidbush;
the laptop checks it with `python -m vidbush.live_probe` (docs/HANDOFF.md section 10f).

### Algrow TTS (https://api.algrow.online), ported, unverified live

- `POST /api/generate-simple`, multipart form `{script, voice_id, model_id}`, `Authorization: Bearer`, returns
  `{success, job_id}`. `GET /api/job-status/{job_id}` until `completed` or `failed`, then `audio_url` (hosted MP3,
  no auth). Same key as Algrow images. New type `algrow_tts`.
- No word timings in the reply: Vidbush aligns the audio against the script (faster-whisper).
- HTTP 400 "Text must be at least 200 characters." below 200 characters: short scripts are padded with
  `<break time="1.0s" />` tags, as RadioPlay does.
- RadioPlay measured about 31 s plus 1 ms per character (2,581 chars in 35 s, 25,019 in 57 s) and trusted a
  20,000 character request; Vidbush chunks are at most 4,500 characters. Poll every 3 s, give up after 600 s.
- RadioPlay ran this seat on eleven_multilingual_v2 and later eleven_v3; eleven_v4 on Algrow is not proven
  anywhere yet. A refusal that names the model switches the run to eleven_multilingual_v2.
- Subscription: recorded as 0 USD with the credit count. The TTS credit field name is not proven (RadioPlay's
  TTS adapter reads `credits` from the job; its image adapter found `credits_used` on the submit reply), so
  Vidbush reads `credits_used`, `credit_cost`, then `credits`. `GET /api/credits` returns
  `{credits: {limit, used, remaining}}` (seen live by RadioPlay for images) and is the probe.

### Anthropic direct (https://api.anthropic.com), ported, unverified live

- RadioPlay calls Anthropic through its OpenAI compatible route (`/v1/chat/completions`, Bearer key) and found:
  `response_format: json_object` refused (400 "Input should be 'json_schema'"), and claude-sonnet-5-5 answered
  400 "`temperature` is deprecated for this model". Vidbush uses the native Messages API instead: JSON is asked
  for in the system prompt and temperature is never sent.
- Model ids on the direct API are dashed: `claude-opus-5-5`, `claude-sonnet-5-5` (RadioPlay item 1299).

### Wellflow cheap key and CloseRouter Anthropic ids, ported, unverified live

- RadioPlay's `GET /models` with the cheap key (2026-10-16) lists `claude-opus-5-5` and `claude-sonnet-5-5`
  (no prices); CloseRouter lists only the dotted `anthropic/claude-opus-5.5` and `anthropic/claude-sonnet-5.5`.
  Vidbush resolves both spellings by family and version.
- Live run 1 (2026-11-04) saw the cheap key answer 403 `model_not_in_channel (cheap)` for claude-sonnet-5, while
  RadioPlay runs claude-opus-5-5 and claude-sonnet-5-5 on it daily. The channel likely allows only the 5.5 ids.
  A model listed by `/models` may still be refused by the channel: `live_probe stream wellflow_cheap` checks it.
- Streaming: RadioPlay's 12 call grid passed 6 of 6 streamed Wellflow calls at 45 to 65 s and failed 5 of 6
  non streamed calls at Wellflow's 240 s edge (502, 503, 504); Wellflow sends `: ping` every few seconds.

### Fragile in RadioPlay's clients (not copied)

- RadioPlay's SSE reader sets `finish_reason` to "stop" when the stream never sent one, so a stream cut before
  its end passes as a whole reply. Vidbush raises TransportError for a stream with no `[DONE]` and no finish
  reason, and the request is retried.
- RadioPlay's SSE reader buffers the whole body before parsing, so its deadline covers the full body but a ping
  stream that never ends is only stopped by the outer timer. Vidbush checks its own deadline on every line.
- RadioPlay's Algrow TTS poll knows only `completed` and `failed`; any other terminal word ("error",
  "cancelled") would poll until the timeout, and an HTTP 200 `{success: false}` reply is not read as an error.
  Vidbush treats those as terminal and maps the envelope to a refusal, a rate limit or a retry.
- The Algrow short script padding relies on break tags being silent. That held on eleven_multilingual_v2;
  on eleven_v4 a tag could be spoken or ignored. Check `live_probe algrow` with a short line if it matters.
