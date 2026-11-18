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

## Seat policy of 2026-11-11 (seats lane), ported, unverified live

Taken from CaseFile's seat notes (seats.yaml and MAIN_NOTES_2026-11-10.md) and RadioPlay; nothing here was sent
live from Vidbush. Check with `python -m vidbush.live_probe seats` and the probes in docs/HANDOFF.md section 10f.

### openlux (https://api.openlux.ai/v1), director and researcher second, editor and vision roles second

- OpenAI compatible: `POST /chat/completions`, `Authorization: Bearer OPENLUX_API_KEY`, base URL from
  `OPENLUX_BASE_URL`. Streamed like the other relays (`options.stream: true`); if it rejects `stream_options` the
  quirk is learned and the stream goes on without the usage request.
- `GET /models` is free, returns about 328 ids including `claude-sonnet-5-5` and `claude-opus-5-5`, and carries
  no prices. Sonnet 5.5 costs 0.1765 USD per million prompt tokens and 0.8824 per million completion tokens
  (CaseFile), recorded as `options.price_per_mtok` on the role entry. One Sonnet request with one image passed.
- Not known: whether openlux lists Grok 4.7 and Gemini 3.8 Flash and under which ids, and whether its Gemini takes
  images. Those entries use `model_match` (grok 4.7, gemini 3.8-flash, `prefix: ""`), so an unlisted family leaves
  the seat unconfigured with one warning and Wellflow serves.

### Grok and Gemini ids

- CloseRouter: `x-ai/grok-4.7`, `google/gemini-3.8-flash`, `anthropic/claude-opus-5.5` (CaseFile). Its Sonnet 5.5
  id is not verified (likely `anthropic/claude-sonnet-5.5`), so director and researcher resolve it by
  `model_match`.
- Wellflow (normal key): `gemini-3.8-flash` verified on 2026-11-04; `grok-4.7` is not verified, so the editor's
  Wellflow entry resolves it by `model_match`.
- `model_match` without the claude prefix matches the bare and vendor spellings (`grok-4.7`, `x-ai/grok-4.7`), and
  never a sibling (`gemini-3.8-flash-lite`, `gemini-3.8-flash-preview`, `grok-4.7-fast`).
- Wellflow cheap key: serves `claude-opus-5-5` and answers 403 `model_not_in_channel` for `claude-sonnet-5-5`
  (CaseFile), so it sits only on the writer. Flat 0.30 USD per million tokens.

### Anthropic direct key

- The key name in Adam's `.env.local` is `ANTHROPIC_KEY`. On 2026-11-11 it answered 400 "You have reached your
  specified API usage limits ... 2026-12-01" (CaseFile). That reply now raises `SeatUnavailable`: the seat is marked
  down for 10 minutes at the first such reply instead of costing a round trip on every request.
- The process environment of a Claude Code session sets `ANTHROPIC_BASE_URL`, and process env wins over the env
  files, so the provider reads `ANTHROPIC_DIRECT_BASE_URL` only.

### Elevate TTS (https://public-api.elevate.uno), ported from RadioPlay, unverified live from Vidbush

- `POST /v2/media` JSON `{type: "tts", prompt, voice_id, model_id}`, `Authorization: Bearer ELEVATE_API_KEY` (the same
  key as Elevate images) returns `{success, data: {id, status}}`. Poll `GET /v2/media/{id}?type=tts` until
  `completed` (terminal: completed, failed, cancelled, canceled, error), then download `data.result_url` (no auth).
  New type `elevate_tts` in `vidbush/providers/tts_elevate.py`.
- Model ids from Elevate's own catalogue (`GET /v2/media/types`, RadioPlay 25/10/2026): `eleven_v4`, `eleven_v3`.
  eleven_v4 takes no voice_settings and no audio tags, so none are sent. No word timings come back: Vidbush aligns
  the audio against the script with faster-whisper.
- Voice: George `JBFqnCBsd6RMkjVDRZzb` is verified on Elevate (and on Algrow); the professional voice Deep Ray is not
  available on Elevate. Brian is not verified there, so the default brand maps elevate_tts to George.
- Timeouts and pacing (RadioPlay's working values): request timeout 400 s plus 0.002 s a character, capped at 600 s;
  poll every 18 s. A 429 `{"code": "RATE_LIMIT_EXCEEDED", "message": "TTS concurrent limit reached (3/3)", "details":
  {"used": 3, "limit": 3}}` is the concurrent limit: waited out, not a rate limit. A plain 429 retries 4 times.
- A task still running at its deadline keeps a lane slot; the retry polls that task instead of submitting again.
- A task that failed with "Voice not available" is the request's fault (a refusal), not an outage.
- Probe: `GET /v2/media/usage` with the Bearer key first (health never rejects a bad key, as the image probe
  found), then `GET /v2/media/health`, `data.tts.status` must be `operational` (RadioPlay saw `degraded` with success_rate
  60.0 on 05/09/2026).

## Live probe of 2026-11-11 (live-run-2, `doctor --probe`)

Every seat in the 2026-11-11 policy answered on this laptop, which verifies the ids listed above as unverified:

- writer: wellflow_cheap `claude-opus-5-5`, closerouter `anthropic/claude-opus-5.5`, anthropic_direct
  `claude-opus-5-5` all ok. The direct key no longer answers "usage limits".
- director and researcher: closerouter `anthropic/claude-sonnet-5.5`, openlux `claude-sonnet-5-5`, anthropic_direct
  `claude-sonnet-5-5` all ok.
- editor: closerouter `x-ai/grok-4.7`, openlux `grok-4.7`, wellflow `grok-4.7` all ok.
- judge, vision, cheap_classifier: closerouter `google/gemini-3.8-flash` and openlux `gemini-3.8-flash` ok. Wellflow
  `gemini-3.8-flash` answered ok for vision in about 56 s, but twice in the same probe sent a stream error event
  `{code: model_unavailable, type: server_error, message: "temporarily unavailable. Please retry in a few
  seconds"}`. Vidbush raises TransportError for any stream error event, so a run retries it; the probe makes one
  attempt only. A second probe a few minutes later (`live_probe stream wellflow --role judge`) answered ok in 21.6 s.
- tts: elevenlabs `eleven_v4` (voice Brian), elevate_tts and algrow_tts (`eleven_v4`, 1125 Algrow credits) ok.
- image: Elevate `gpt-image-2.5-fast` and Algrow `gpt-image-2.5` ok.
- video and search have no configured provider (as in live run 1); no stage of the Alaska run calls them.

## V2 sources (2026-11-17, parallel3 lane, branch v2/sources)

Live checks from this laptop. Keys are read by name through settings.key; no value is ever printed.

### Web image search (`webimage`, vidbush/source/webimage.py)

- Provider order: SERPER_API_KEY (POST google.serper.dev/images), then FIRECRAWL_API_KEY (POST
  api.firecrawl.dev/v2/search with `sources: [{type: images}]`), then the search seat (SEARCH_PROVIDER or the
  web_search seat type: serper, brave or tavily, key SEARCH_API_KEY). One provider per run.
- Live: only Firecrawl. SERPER_API_KEY is not set. SEARCH_API_KEY (tavily) answers 401 "missing or invalid API
  key" with the key in the body and as a Bearer header, so the research seat's web search is broken too.
  Serper, Brave and Tavily parsers are tested on hand-built fixtures from their documented shapes only.
- Firecrawl images: 2 credits per search; each hit has title, imageUrl, imageWidth, imageHeight, url (the
  page) and position. Hits are often facebook.com (lookaside.fbsbx.com), reddit (preview.redd.it), alamy and
  worthpoint previews; many hosts serve WEBP or AVIF at .jpg URLs or an HTML page on hotlink. fetch decodes
  every file with PIL, accepts JPEG, MPO, PNG, WEBP and TIFF only, and rejects with a logged reason
  otherwise. Requests send Accept image/* and Referer = page URL.
- Licence is always "unknown (web image search)", flagged licence_unknown; author is the page host.
- The classes it serves are web_image and newspaper. Encyclopedic (bare name) queries are not sent to it: on
  the open web a bare name returns its namesakes first ("Balto" gave the 1995 film's Blu-ray).

### Openverse (`openverse`, vidbush/source/openverse.py)

- Anonymous GET api.openverse.org/v1/images/?q=&page_size=&mature=false; no key. The anonymous limit is small
  (documented as roughly 100 requests a day and a few a minute; not measured). Host rate 1 per 4 s, burst 3;
  run budget 90 searches. OAuth client credentials (more quota) are not implemented.
- Licence codes map to canonical names (pdm to Public domain, cc0 to CC0 1.0, by-sa 2.0 to CC BY-SA 2.0);
  Openverse's own attribution line is the credit. Many CC BY-NC and ND results (flagged nc_nd, never hidden).
- The image URL is often Flickr's medium rendition (1024 px or less), so results are often low_res.
- fetch does not check the body decodes: a dead Flickr link could arrive as a placeholder image.

### Chronicling America (`chronam`, vidbush/source/chronam.py)

- The chroniclingamerica.loc.gov API is gone (308 to www.loc.gov). Search uses the loc.gov advanced form from
  the Library's notebooks: /collections/chronicling-america/?dl=page&searchType=advanced&ops=PHRASE&qs=...
  &start_date=&end_date=&c=&fo=json. The docs ask to keep result sets under 100,000 hits.
- Live: www.loc.gov answers a Cloudflare 403 "Just a moment" challenge to every search and item request from
  this laptop (with the contact User-Agent and Accept: application/json too). Search is unusable from this
  connection; it fails in under a second and pauses the host (see Library of Congress below).
- tile.loc.gov is not challenged: IIIF page images (about 5800x7400 native; /full/2000,/ in about 8 s cold)
  and the word coordinates service (JSON even with format=alto_xml; first calls can time out) work. A real
  page fetched live gave a readable 1754x986 headline crop and the 2000 px page.
- Fixtures: chronam_search.json is the documented notebook result (real, trimmed) plus two synthetic results;
  chronam_coords.json is a real tile.loc.gov response.

### Wikimedia by entity (vidbush/source/wikimedia.py) and Wikidata

- wbsearchentities ranks by its own order, not fame ("Balto": a Finnish research vessel first). Items often
  keep their name only under language "mul"; the adapter asks languages=en|mul. P373 holds the bare category
  name. Commons imageinfo URLs now carry ?utm_ parameters, stripped for dedupe.
- About 4 requests per entity search (cap 5), then up to 2 for the text fallback. A Wikidata outage falls
  through to text search. For video the entity category rarely holds video, and the text fallback uses the
  query, not the entity label, so it can return namesakes (NASA hurricane videos for "Hurricane Turn").

### Library of Congress (`loc`)

- Unusable from this connection: every www.loc.gov request is a Cloudflare challenge, with or without the
  operator contact User-Agent. The challenge pause is now 600 s (was 3900 s) and requests during it are
  refused at once, so it never stalls a stage; the run budget caps retries.

### Segment chooser (vidbush/source/segment.py)

- Uses the cheap_classifier seat (Gemini 3.8 Flash on CloseRouter): one call per video and subject with one
  numbered 5 x 4 sheet (1280 x 576), detail high. Cached under <source_cache>/segments/.
- YouTube storyboards: yt-dlp lists them as formats with format_note "storyboard" (mhtml, fragments are JPEG
  sprite sheets of rows x columns tiles). No video is downloaded to read them.
- Proof run (2026-11-17): 22 segment choices, 12 named a frame, 10 found none (the caller kept its old rule);
  22 calls cost 0.0064 USD in all on CloseRouter.
- YouTube: a bot check ("Sign in to confirm you're not a bot") came after four proof passes in about three
  hours without cookies; the adapter paused YouTube for 30 minutes as designed.
## Elevate catalogue, image and video (2026-11-15, run 3 art lane, read live with Adam's key)

Source: `GET /v2/media/types`. The catalogue is the model list; there is no model endpoint (`GET /v2/media/models`
answers 422 because `models` is parsed as a task id, `GET /v2/models` answers 404; the Elevate MCP
`get_media_types` returns the same text). Credits per image were measured from `GET /v2/media/usage`
`image.hourly.credits_used` before and after one call each, while no other work used the key. Raw replies:
`E:\coding projects\Vidbush\review_run3\catalogue\`.

Image models (`POST /v2/media {type: image, model, prompt, aspect_ratio, seed, params, reference_images}`):

| id | reference images | credits | time | output (landscape) | notes |
|---|---|---|---|---|---|
| `nano_banana_pro` | up to 10, seed | 5 (measured) | about 41 s | 1376x768 PNG | best map painter: keeps the reference coast, no watermark seen; the art layer's map and illustration model |
| `nano_banana_2` | up to 10 | 5 (measured) | about 40 s | 1376x768 | close to pro; drifted the Yukon delta coast in the probe |
| `gemini_flash_image` | 1 (switches to edit mode) | 3 (measured) | about 32 s | 1365x768 | registered well; the art layer's map fallback |
| `gpt-image-2.5-fast` | up to 10 | 4 (catalogue and measured) | about 39 s | 1672x941 with a reference | reframed the map and copied the reference's border line, so it fails registration |
| `gpt-image-2.5-thinking` | up to 10 | 7 (catalogue) | about 70 s | not probed | |
| `Batman` | none (text only) | 0 (launch promo) | about 6 s | | `params.style`: default, portrait, ghibli, anime, cinematic, 3d, pixel, oil, sketch, sticker |

- `params.resolution: "2K"` is ignored by `nano_banana_pro` (still 1376x768); painted bases are upscaled at render time.
- No sparkle watermark was visible on the nano_banana and gemini outputs, so painted maps use no edge trim (the
  image seat's `edge_trim: 0.05` is for gpt-image).

Video models (`POST /v2/media {type: video, model, prompt, operation, source_image, aspect_ratio, params}`):

| id | operations | credits | duration | output | time |
|---|---|---|---|---|---|
| `veo_3_1` (Veo 3.1) | `prompt` (text to video) and `image` (image to video, `source_image` as a base64 data URL) | 720p 1, 1080p 2 (default) | 7.93 s whatever is asked (no duration field) | 1280x720 at 720p, 24 fps, H.264 with an AAC audio track | 363 s for one 720p clip on 2026-11-15 |

- It is the only video model listed. Choice: `veo_3_1` at 1080p. Runner-up and fallback: `veo_3_1` at 720p (half
  the credits). fal stays configured second in the video role when a key is set.
- The first frame of an image to video clip is the source still. Veo added clouds and a slight camera drift to a
  painted map despite "no camera move" in the prompt. The art layer's map motion line now asks for sea ripples
  and light only, and a map clip must pass registration on its last frame.
- Limits on this key (`GET /v2/media/usage`): image 3 at once, 200 tasks and 800 credits an hour; video 1 at
  once, 40 clips and 80 credits an hour, no daily cap (`daily: null`); tts 3 at once. Budgets reset at the top
  of the UTC hour (`resets_at`) and are shared with the dashboard and every other session on the key. Image and
  video credits are separate budgets.
- Adapter: `vidbush/providers/video_elevate.py`, provider type `elevate_video`, registered in
  `seats.VIDEO_TYPES` and in `doctor --probe` (usage with the key, then `data.video.status` must be operational).
  It submits, polls every 10 s for up to 900 s, and downloads without auth. The task id is kept beside the output
  (`<out>.task.json`), so a timeout or a restart resumes the running task instead of paying twice. The
  concurrent limit is waited out (usage before the submit, the capacity 429 during it). A refused task is never
  resubmitted.
- In the proving run veo moved the camera (3 to 5 percent zoom or drift) on all 5 painted map clips and added objects or morphing to 4 of 8 photo-style AI stills; 6 of 10 illustration clips passed every check. Throughput was about one clip every 6 to 8 minutes on the one-at-a-time lane.
- Images with references for the art layer: `vidbush/art/elevate.py` (RadioPlay's `reference_images` shape),
  through the image seat's own client, envelope and polling.

## video-download-api.com (paid YouTube route, night 2 integration)

- Host `https://p.savenow.to`. Key `VIDEO_DOWNLOAD_API_KEY` (also read as `VDA_API_KEY`), sent as the query
  parameter `apikey` (download) or `api` (info, subtitles); never a header. Client: `vidbush/source/youtube_api.py`.
- `GET /ajax/download.php?url=&format=720&apikey=&add_info=1&allow_extended_duration=0` returns `{success, id,
  info}`; poll `GET /ajax/progress.php?id=` until `progress` reaches 1000, then use `download_url` only when
  `success` is 1 or true. `success` 0 below 1000 is normal (still running).
- `GET /ajax/api.php?function=i&api=&u=` answers title, duration, thumbnail directly (no job). Not used for
  licence: the service reports none.
- `GET /ajax/subtitles?url=&api=&lang=en&format=vtt` returns `{id, progress_url}`; poll progress_url the same way.
- Prices (pricing page, 2026-11-18): MP4 360p, 720p, 1080p 0.0002 USD per job (180 minute base duration), 1440p
  0.0003, 4K 0.00035; VTT or SRT 0.0001. Charged when a job is accepted and returns an id, before the file is
  ready; a request rejected first (bad key: HTTP 401 invalid_api_key, no balance, too long) is not charged.
- Whole videos only, no time sections: Vidbush fetches 720p whole and cuts windows locally.
- Measured here (2 live videos, sled dog footage): a 4 minute video took about 100 s from request to file
  (queue plus download); two of five Creative Commons results were 480p at best and are rejected after download
  (the 0.0002 USD is spent: the service shows no formats beforehand); one file link was cut off mid download
  ("Response ended prematurely"), now fetched once more on the same paid job.
