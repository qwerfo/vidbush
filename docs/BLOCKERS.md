# Blockers

Subsystems that failed three times, with the exact error and what to try.

- Network: the build sandbox proxy refuses connections to en.wikipedia.org, commons.wikimedia.org, archive.org, loc.gov, images-api.nasa.gov, youtube.com, api.openai.com, api.elevenlabs.io and huggingface.co ("connect_rejected"). Adapters are implemented and unit tested with canned responses. To verify live: run `python -m vidbush doctor` and a real project on a machine with internet.
