# Blockers

Subsystems that failed three times, with the exact error and what to try.

- Network: the build sandbox proxy refuses connections to en.wikipedia.org, commons.wikimedia.org, archive.org, loc.gov, images-api.nasa.gov, youtube.com, api.openai.com, api.elevenlabs.io and huggingface.co ("connect_rejected"). Adapters are implemented and unit tested with canned responses. To verify live: run `python -m vidbush doctor` and a real project on a machine with internet.
- faster-whisper model download (huggingface.co) is blocked here, so voiceover transcription without a script was not run live; voiceover with a script uses silence-based alignment and passes A3.
- deb.debian.org is blocked here, so the Docker image uses an ubuntu:24.04 base (archive.ubuntu.com is reachable). Docker Hub rate limited pulls (429) once; retrying after a few minutes worked.
- DepthFlow is not installable here; the integration is wired but untested and off by default.
