# Blockers

Subsystems that failed three times, with the exact error and what to try.

- Network: the build sandbox proxy refuses connections to en.wikipedia.org, commons.wikimedia.org, archive.org, loc.gov, images-api.nasa.gov, youtube.com, api.openai.com, api.elevenlabs.io and huggingface.co ("connect_rejected"). Adapters are implemented and unit tested with canned responses. To verify live: run `python -m vidbush doctor` and a real project on a machine with internet.
- faster-whisper model download (huggingface.co) is blocked here, so voiceover transcription without a script was not run live; voiceover with a script uses silence-based alignment and passes A3.
- deb.debian.org is blocked here, so the Docker image uses an ubuntu:24.04 base (archive.ubuntu.com is reachable). Docker Hub rate limited pulls (429) once; retrying after a few minutes worked.
- DepthFlow is not installable here; the integration is wired but untested and off by default.

## Live test (internet access granted after the build)

- Working live: Internet Archive, Library of Congress and NASA search plus download with licences; YouTube search; faster-whisper model download and transcription (voiceover with no script ran end to end, 601 words in 30 s).
- YouTube downloads: "Sign in to confirm you're not a bot" from this cloud IP, and section downloads then fail with ffmpeg exit code 8. Expect the same on a VPS. Fix: set YOUTUBE_COOKIES (cookies.txt from a browser logged into a throwaway Google account) or YOUTUBE_COOKIES_FROM_BROWSER. Vidbush now stops trying YouTube for the rest of a run after the first bot check and logs this fix.
- Wikipedia and upload.wikimedia.org: 429 for agents without contact info (fixed: the User-Agent now carries the project URL, and VIDBUSH_CONTACT adds yours). After repeated tests this shared cloud IP was rate limited regardless of agent, so full live research was not observed here.
- faster-whisper 1.2.1 breaks with PyAV 17 or newer (unexpected keyword argument metadata_errors). av is pinned to 16.1.0.
