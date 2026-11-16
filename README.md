# Automatic-Short-form-video-generator
A program that, simply by inputting a script (.txt) and an audio file (.mp3/.wav), can generate a short video with images that are either embedded or generated locally, depending on your configuration.

## GTA 6 clipper skill

`.claude/skills/gta6-clip` is a Claude Code skill that turns a raw GTA 6 recording into
captioned 9:16 Shorts. Open this repo in Claude Code, drop a video and say "clip this".

Needs ffmpeg and Python 3.9+. First run: `python3 .claude/skills/gta6-clip/scripts/clipper.py setup`.
