#!/usr/bin/env python3
"""GTA 6 clipper: raw gameplay/stream video -> captioned 9:16 Shorts.

Subcommands
  setup    install the speech model (one time, ~650 MB)
  analyze  score every second, transcribe, list clip candidates + contact sheets
  frame    export one frame with a 10x10 grid (to locate a facecam)
  render   cut, reframe, caption and export the clips described in a plan.json
  check    verify exported clips and lint hooks

Needs: ffmpeg/ffprobe, Python 3.9+, numpy, Pillow. Captions also need sherpa-onnx.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = SKILL_DIR / "assets" / "fonts"
CACHE = Path(os.environ.get("GTA6_CLIP_CACHE", Path.home() / ".cache" / "gta6-clip"))
MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

OUT_W, OUT_H = 1080, 1920

# Signal weights for the per-second score (sum to 1).
WEIGHTS = {"loud": 0.30, "jump": 0.25, "keywords": 0.20, "motion": 0.15, "speech": 0.10}

HYPE = [
    # English
    "no way", "oh my god", "omg", "what the", "wtf", "let's go", "lets go", "holy", "insane",
    "crazy", "bro", "dude", "run", "wait", "are you serious", "shut up", "unbelievable",
    "clip that", "dead", "help", "cops", "police", "wanted", "crash", "exploded", "glitch",
    # French
    "non", "mais non", "oh la la", "putain", "c'est pas possible", "pas possible", "incroyable",
    "attends", "attend", "quoi", "wesh", "frère", "frere", "sérieux", "serieux", "les flics",
    "la police", "cours", "aide", "dingue", "ouf", "malade", "c'est quoi", "bug",
]
SCENARIO_CUES = [
    "let me show you", "try this", "first time", "welcome to", "this is my", "watch this",
    "je vais vous montrer", "regardez", "première fois", "premiere fois", "on va essayer",
    "bienvenue", "on teste",
]
SWEARS = {
    "fuck": "F*CK", "fucking": "F*CKING", "fucked": "F*CKED", "shit": "SH*T", "bitch": "B*TCH",
    "motherfucker": "MOTHERF*CKER", "asshole": "A**HOLE", "dick": "D*CK", "pussy": "P*SSY",
    "putain": "P*TAIN", "merde": "M*RDE", "connard": "C*NNARD", "salope": "S*LOPE",
    "enculé": "ENC*LÉ", "encule": "ENC*LE", "bordel": "B*RDEL", "nique": "N*QUE", "pute": "P*TE",
}
VAGUE_HOOKS = [
    "wait for it", "watch this", "you won't believe", "watch till the end", "watch until the end",
    "attends la fin", "attendez la fin", "regarde ça", "regarde ca", "regardez ça",
    "tu vas pas le croire", "vous n'allez pas le croire", "incroyable",
]


def die(msg: str) -> None:
    sys.exit(f"error: {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        die(f"command failed: {' '.join(map(str, cmd[:6]))} ...\n{p.stderr[-1500:]}")
    return p


def probe(path: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    info = json.loads(p.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        die(f"{path} has no video stream")
    num, den = (v.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        rot = int(sd.get("rotation", 0) or 0)
    if abs(rot) in (90, 270):
        w, h = h, w
    return {
        "width": w, "height": h, "fps": fps or 30.0, "has_audio": a is not None,
        "duration": float(info["format"].get("duration") or v.get("duration") or 0),
    }


def fmt_t(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


# --------------------------------------------------------------------------- setup

def model_ready() -> bool:
    return (CACHE / MODEL_NAME / "encoder.int8.onnx").exists() and (CACHE / "silero_vad.onnx").exists()


def cmd_setup(_args) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found. Install ffmpeg first.")
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        print("installing sherpa-onnx ...")
        run([sys.executable, "-m", "pip", "install", "-q", "sherpa-onnx", "numpy", "Pillow"])
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "silero_vad.onnx").exists():
        print("downloading voice detector ...")
        urllib.request.urlretrieve(VAD_URL, CACHE / "silero_vad.onnx")
    if not (CACHE / MODEL_NAME / "encoder.int8.onnx").exists():
        print("downloading speech model (~650 MB) ...")
        tmp = CACHE / "model.tar.bz2"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        with tarfile.open(tmp) as tf:
            tf.extractall(CACHE)
        tmp.unlink()
    print(f"ready: {CACHE}")


# ------------------------------------------------------------------------- analyze

def extract_wav(video: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)])


def load_wav(wav: Path):
    import numpy as np
    import wave
    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(samples) -> list[dict]:
    """Return [{w, s, e}] word list with times in seconds."""
    import sherpa_onnx
    m = CACHE / MODEL_NAME
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(m / "encoder.int8.onnx"), decoder=str(m / "decoder.int8.onnx"),
        joiner=str(m / "joiner.int8.onnx"), tokens=str(m / "tokens.txt"),
        model_type="nemo_transducer", num_threads=os.cpu_count() or 4)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(CACHE / "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = 0.35
    cfg.silero_vad.threshold = 0.35
    cfg.silero_vad.max_speech_duration = 20.0
    cfg.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    win = cfg.silero_vad.window_size
    words: list[dict] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front
            t0 = seg.start / 16000.0
            st = rec.create_stream()
            st.accept_waveform(16000, seg.samples)
            rec.decode_stream(st)
            r = st.result
            seg_end = t0 + len(seg.samples) / 16000.0
            cur = None
            for tok, ts in zip(r.tokens, r.timestamps):
                starts = tok.startswith(" ") or tok.startswith("▁")
                piece = tok.replace("▁", " ").strip()
                if not piece:
                    continue
                if cur is None or (starts and not re.fullmatch(r"[.,!?;:%]+", piece)):
                    if cur:
                        words.append(cur)
                    cur = {"w": piece, "s": round(t0 + ts, 2), "e": round(t0 + ts + 0.2, 2)}
                else:
                    cur["w"] += piece
                    cur["e"] = round(t0 + ts + 0.2, 2)
            if cur:
                words.append(cur)
            for a, b in zip(words[:-1], words[1:]):  # a word ends where the next one starts
                if b["s"] > a["s"]:
                    a["e"] = round(min(max(a["e"], a["s"] + 0.12), b["s"]), 2)
            if words:
                words[-1]["e"] = round(min(words[-1]["e"] + 0.15, seg_end), 2)
            vad.pop()

    for i in range(0, len(samples) - win + 1, win):
        vad.accept_waveform(samples[i:i + win])
        drain()
    vad.flush()
    drain()
    return words


def motion_scores(video: Path, n: int):
    """Scene-change score per second, from keyframes only (fast on long videos)."""
    import numpy as np
    out = np.zeros(n, dtype=np.float32)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", str(video), "-an",
         "-vf", "scale=160:90,select='gte(scene,0)',metadata=print:file=-", "-vsync", "vfr",
         "-f", "null", "-"], capture_output=True, text=True)
    t = None
    for line in p.stdout.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"scene_score=([\d.]+)", line)
        if m and t is not None and int(t) < n:
            out[int(t)] = max(out[int(t)], float(m.group(1)))
    return out


def norm(x):
    import numpy as np
    lo, hi = np.percentile(x, 50), np.percentile(x, 99.5)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def contact_sheet(video: Path, start: float, end: float, out: Path, n: int = 6) -> None:
    from PIL import Image, ImageDraw
    tiles = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                            "-vf", "scale=480:-2", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                           capture_output=True)
        if not p.stdout:
            continue
        import io
        im = Image.open(io.BytesIO(p.stdout)).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 118, 22], fill=(0, 0, 0))
        d.text((5, 5), fmt_t(t), fill=(255, 255, 0))
        tiles.append(im)
    if not tiles:
        return
    w, h = tiles[0].size
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (w * cols, h * rows))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=82)


def cmd_analyze(args) -> None:
    import numpy as np
    video = Path(args.video).resolve()
    if not video.exists():
        die(f"no such file: {video}")
    work = Path(args.work or f"work/{video.stem}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    n = max(1, int(math.ceil(meta["duration"])))
    print(f"source {meta['width']}x{meta['height']} {meta['fps']:.0f}fps {fmt_t(meta['duration'])}")

    words: list[dict] = []
    loud = np.zeros(n, dtype=np.float32)
    if meta["has_audio"]:
        wav = work / "audio.wav"
        extract_wav(video, wav)
        samples = load_wav(wav)
        for i in range(min(n, len(samples) // 16000 + 1)):
            chunk = samples[i * 16000:(i + 1) * 16000]
            if len(chunk):
                loud[i] = 20 * math.log10(float(np.sqrt(np.mean(chunk ** 2))) + 1e-6)
        if args.no_captions:
            print("transcription skipped (--no-captions)")
        elif not model_ready():
            print("WARNING: speech model missing, no transcript. Run: clipper.py setup")
        else:
            print("transcribing ...")
            words = transcribe(samples)
            print(f"{len(words)} words")
        wav.unlink()
    else:
        loud[:] = -100
        print("WARNING: source has no audio track")

    # loudness above the video's own normal level, and sudden rises
    f_loud = norm(loud)
    prev = np.array([loud[max(0, i - 5):i].mean() if i else loud[0] for i in range(n)])
    f_jump = norm(np.clip(loud - prev, 0, None))
    f_motion = norm(motion_scores(video, n)) if not args.no_motion else np.zeros(n)
    kw = np.zeros(n, dtype=np.float32)
    rate = np.zeros(n, dtype=np.float32)
    text_by_sec: dict[int, list[str]] = {}
    for w in words:
        s = int(w["s"])
        if s < n:
            rate[s] += 1
            text_by_sec.setdefault(s, []).append(w["w"].lower())
    for s in range(n):
        ctx = " ".join(sum((text_by_sec.get(k, []) for k in (s - 1, s, s + 1)), []))
        ctx = " " + re.sub(r"[^\w' ]", " ", ctx) + " "
        kw[s] = sum(1 for h in HYPE if f" {h} " in ctx)
    f_kw = np.clip(kw / 3.0, 0, 1)
    f_rate = norm(rate) if rate.any() else rate
    score = (WEIGHTS["loud"] * f_loud + WEIGHTS["jump"] * f_jump + WEIGHTS["keywords"] * f_kw
             + WEIGHTS["motion"] * f_motion + WEIGHTS["speech"] * f_rate)
    smooth = np.convolve(score, np.ones(3) / 3, mode="same")

    target = args.target
    want = args.max or int(min(25, max(3, round(meta["duration"] / 180))))
    order = np.argsort(-smooth)
    peaks: list[int] = []
    for i in order:
        if smooth[i] <= 0 or (peaks and smooth[i] < 0.25 * smooth[peaks[0]]):
            break
        if all(abs(int(i) - p) >= target for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= want:
            break

    word_starts = [w["s"] for w in words]
    word_ends = [w["e"] for w in words]

    def snap(t: float, pool: list[float], back: bool) -> float:
        near = [x for x in pool if abs(x - t) <= 2.0]
        if not near:
            return t
        best = min(near, key=lambda x: abs(x - t))
        return best - 0.15 if back else best + 0.25

    cands = []
    for rank, p in enumerate(peaks, 1):
        start = max(0.0, p - 0.6 * target)
        end = min(meta["duration"], start + target)
        start = max(0.0, snap(start, word_starts, True))
        end = min(meta["duration"], snap(end, word_ends, False))
        text = " ".join(w["w"] for w in words if start <= w["s"] < end)
        sig = {"loud": float(f_loud[p]), "jump": float(f_jump[p]), "keywords": float(f_kw[p]),
               "motion": float(f_motion[p]), "speech": float(f_rate[p])}
        cands.append({"id": rank, "kind": "peak", "peak": p, "start": round(start, 2), "end": round(end, 2),
                      "score": round(float(smooth[p]), 3), "why": max(sig, key=sig.get),
                      "signals": {k: round(v, 2) for k, v in sig.items()}, "text": text})
    # transcript-cued moments that have no loudness spike
    full = [(w["s"], w["w"].lower()) for w in words]
    seen = [c["peak"] for c in cands]
    for i in range(len(full)):
        phrase = " ".join(x[1] for x in full[i:i + 5])
        cue = next((c for c in SCENARIO_CUES if phrase.startswith(c)), None)
        t = full[i][0]
        if cue and all(abs(t - s) >= target for s in seen) and len(cands) < want + 8:
            seen.append(t)
            end = min(meta["duration"], t + target)
            cands.append({"id": len(cands) + 1, "kind": "scenario", "peak": int(t), "start": round(max(0, t - 0.5), 2),
                          "end": round(end, 2), "score": round(float(smooth[min(n - 1, int(t))]), 3),
                          "why": f'cue "{cue}"', "signals": {},
                          "text": " ".join(w["w"] for w in words if t - 0.5 <= w["s"] < end)})

    (work / "transcript.json").write_text(json.dumps(words, ensure_ascii=False))
    (work / "scores.json").write_text(json.dumps([round(float(x), 3) for x in smooth]))
    (work / "analysis.json").write_text(json.dumps(
        {"source": str(video), "meta": meta, "candidates": cands}, ensure_ascii=False, indent=1))
    sheets = work / "sheets"
    sheets.mkdir(exist_ok=True)
    lines = [f"# Candidates for {video.name}", "",
             f"Source {meta['width']}x{meta['height']}, {fmt_t(meta['duration'])}, "
             f"{len(words)} words transcribed.", ""]
    for c in cands:
        contact_sheet(video, c["start"], c["end"], sheets / f"cand_{c['id']:02d}.jpg")
        lines += [f"## {c['id']}. {fmt_t(c['start'])} -> {fmt_t(c['end'])}  "
                  f"({c['end'] - c['start']:.0f}s, score {c['score']}, {c['kind']}: {c['why']})",
                  f"sheet: sheets/cand_{c['id']:02d}.jpg", "", f"> {c['text'] or '(no speech)'}", ""]
    (work / "candidates.md").write_text("\n".join(lines))
    print(f"{len(cands)} candidates -> {work / 'candidates.md'}")


# --------------------------------------------------------------------------- frame

def cmd_frame(args) -> None:
    from PIL import Image, ImageDraw
    import io
    video = Path(args.video)
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(args.time), "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=1280:-2", "-f", "image2pipe", "-c:v", "mjpeg", "-"], capture_output=True)
    if not p.stdout:
        die("could not read a frame at that time")
    im = Image.open(io.BytesIO(p.stdout)).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for i in range(1, 10):
        d.line([(w * i / 10, 0), (w * i / 10, h)], fill=(255, 255, 0), width=1)
        d.line([(0, h * i / 10), (w, h * i / 10)], fill=(255, 255, 0), width=1)
        d.text((w * i / 10 + 3, 3), f"x={i / 10:.1f}", fill=(255, 255, 0))
        d.text((3, h * i / 10 + 3), f"y={i / 10:.1f}", fill=(255, 255, 0))
    out = Path(args.out or "frame_grid.jpg")
    im.save(out, quality=88)
    print(out)


# -------------------------------------------------------------------------- render

def ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def ass_escape(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def censor(word: str) -> str:
    core = re.sub(r"[^\w]", "", word.lower())
    if core in SWEARS:
        return re.sub(re.escape(core), SWEARS[core], word, flags=re.I)
    return word


def caption_chunks(words: list[dict], max_words: int = 3, max_chars: int = 14) -> list[list[dict]]:
    chunks, cur = [], []
    for w in words:
        chars = sum(len(x["w"]) for x in cur) + len(cur) + len(w["w"])
        gap = cur and w["s"] - cur[-1]["e"] > 0.45
        if cur and (len(cur) >= max_words or chars > max_chars or gap or re.search(r"[.!?]$", cur[-1]["w"])):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def wrap_hook(text: str, width: int = 16) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def build_ass(clip: dict, words: list[dict], dur: float, caption_y: int, hook_y: int) -> str:
    yellow, white = "&H0000F0FF&", "&H00FFFFFF&"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Montserrat Black,96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,8,3,5,40,40,40,1
Style: Hook,Montserrat Black,74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,7,3,5,40,40,40,1
Style: HookBox,Montserrat Black,66,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,14,0,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    hook = (clip.get("hook") or "").strip()
    if hook:
        style = "HookBox" if clip.get("hook_style") == "box" else "Hook"
        accent = "&H00D03090&" if style == "HookBox" else yellow
        base = "&H00000000&" if style == "HookBox" else white
        keys = {re.sub(r"[^\w]", "", k) for k in (clip.get("hook_keyword") or "").lower().split()}
        lines = wrap_hook(hook)
        size = 86 if style == "Hook" else 74
        longest = max(len(l) for l in lines)
        if longest > 14:
            size = int(size * 14 / longest)
        step = int(size * 1.32)
        hook_end = min(dur, float(clip.get("hook_seconds", 4.0)))
        for i, line in enumerate(lines):
            parts = []
            for word in line.split():
                w = ass_escape(word)
                if re.sub(r"[^\w]", "", word.lower()) in keys:
                    parts.append(f"{{\\c{accent}}}{w}{{\\c{base}}}")
                else:
                    parts.append(w)
            y = hook_y + i * step
            ev.append(f"Dialogue: 2,{ass_time(0)},{ass_time(hook_end)},{style},,0,0,0,,"
                      f"{{\\an5\\pos({OUT_W // 2},{y})\\fs{size}\\fad(0,150)}}{' '.join(parts)}")
    if clip.get("captions", True):
        chunks = caption_chunks(words)
        for ci, chunk in enumerate(chunks):
            shown = [ass_escape(censor(w["w"]).upper()) for w in chunk]
            chars = sum(len(s) for s in shown) + len(shown) - 1
            size = 96 if chars <= 11 else max(60, int(96 * 11 / chars))
            nxt = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else dur
            for wi, w in enumerate(chunk):
                s = w["s"]
                e = chunk[wi + 1]["s"] if wi + 1 < len(chunk) else min(nxt, w["e"] + 0.5)
                e = min(e, dur)
                if e - s < 0.04:
                    continue
                text = " ".join(f"{{\\c{yellow}}}{t}{{\\c{white}}}" if k == wi else t for k, t in enumerate(shown))
                pop = "\\fscx88\\fscy88\\t(0,70,\\fscx100\\fscy100)" if wi == 0 else ""
                ev.append(f"Dialogue: 1,{ass_time(s)},{ass_time(e)},Cap,,0,0,0,,"
                          f"{{\\an5\\pos({OUT_W // 2},{caption_y})\\fs{size}{pop}}}{text}")
    return head + "\n".join(ev) + "\n"


def layout_filter(clip: dict, meta: dict) -> tuple[str, int, int]:
    """Return (filter producing [v] from [0:v], caption_y, hook_y)."""
    layout = clip.get("layout", "focus")
    cx = min(1.0, max(0.0, float(clip.get("crop_x", 0.5))))
    blur = ("scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,"
            f"scale={OUT_W}:{OUT_H},eq=brightness=-0.12")
    if layout == "crop":       # full-height 9:16 slice of the frame
        return (f"[0:v]crop='min(iw,ih*9/16)':ih:'(iw-min(iw,ih*9/16))*{cx}':0,"
                f"scale={OUT_W}:{OUT_H}[v]", 1300, 330)
    if layout == "fit":        # whole frame on a blurred background
        return (f"[0:v]split=2[a][b];[a]{blur}[bg];[b]scale={OUT_W}:-2[fg];"
                f"[bg][fg]overlay=0:(H-h)/2[v]", 1400, 480)
    if layout == "split":      # facecam on top, gameplay below
        fc = clip.get("facecam")
        if not fc:
            die(f"clip {clip.get('id')}: layout 'split' needs facecam {{x,y,w,h}} (fractions, see `frame`)")
        top_h = 640
        return (f"[0:v]split=2[a][b];"
                f"[a]crop=iw*{fc['w']}:ih*{fc['h']}:iw*{fc['x']}:ih*{fc['y']},"
                f"scale={OUT_W}:{top_h}:force_original_aspect_ratio=increase,crop={OUT_W}:{top_h}[top];"
                f"[b]crop='min(iw,ih*{OUT_W}/{OUT_H - top_h})':ih:'(iw-min(iw,ih*{OUT_W}/{OUT_H - top_h}))*{cx}':0,"
                f"scale={OUT_W}:{OUT_H - top_h}[bot];[top][bot]vstack[v]", 690, 120)
    if layout != "focus":
        die(f"clip {clip.get('id')}: unknown layout '{layout}' (focus, crop, fit, split)")
    # focus: 4:5 slice of the gameplay over a blurred copy, room for hook and captions
    return (f"[0:v]split=2[a][b];[a]{blur}[bg];"
            f"[b]crop='min(iw,ih*4/5)':ih:'(iw-min(iw,ih*4/5))*{cx}':0,scale={OUT_W}:1350[fg];"
            f"[bg][fg]overlay=0:285[v]", 1330, 170)


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\- ]", "", s, flags=re.UNICODE).strip().replace(" ", "_")
    return s[:60] or "clip"


def cmd_render(args) -> None:
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    video = Path(plan["source"]).resolve()
    if not video.exists():
        die(f"source not found: {video}")
    meta = probe(video)
    out_dir = Path(plan.get("out_dir") or plan_path.parent / "exports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = plan_path.parent / "transcript.json"
    all_words = json.loads(tr.read_text()) if tr.exists() else []
    tmp = plan_path.parent / "render_tmp"
    tmp.mkdir(exist_ok=True)
    shutil.copy(FONT_DIR / "Montserrat-Black.ttf", tmp / "Montserrat-Black.ttf")
    fps = min(60, round(meta["fps"]) or 30)
    done = []
    for clip in plan["clips"]:
        start, end = float(clip["start"]), min(float(clip["end"]), meta["duration"])
        dur = end - start
        if dur <= 0:
            die(f"clip {clip.get('id')}: end must be after start")
        words = [{"w": w["w"], "s": round(max(0.0, w["s"] - start), 2), "e": round(min(dur, w["e"] - start), 2)}
                 for w in all_words if start - 0.05 <= w["s"] < end - 0.1]
        if clip.get("words"):   # hand-corrected captions, times relative to the clip
            words = clip["words"]
        vf, cap_y, hook_y = layout_filter(clip, meta)
        cap_y = int(clip["caption_y"] * OUT_H) if clip.get("caption_y") else cap_y
        hook_y = int(clip["hook_y"] * OUT_H) if clip.get("hook_y") else hook_y
        ass = tmp / f"clip_{clip['id']}.ass"
        ass.write_text(build_ass(clip, words, dur, cap_y, hook_y), encoding="utf-8")
        graph = f"{vf};[v]fps={fps},subtitles={ass.name}:fontsdir=.,format=yuv420p[vout]"
        name = f"{int(clip['id']):02d}_{safe_name(clip.get('title') or clip.get('hook') or 'clip')}.mp4"
        out = out_dir / name
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video)]
        if meta["has_audio"]:
            graph += ";[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]"
        else:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", "[aout]" if meta["has_audio"] else "1:a",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{dur:.3f}", str(out)]
        run(cmd, cwd=tmp)
        print(f"rendered {out}  ({dur:.1f}s, {clip.get('layout', 'focus')}, {len(words)} caption words)")
        done.append(str(out))
    (out_dir / "rendered.json").write_text(json.dumps({"plan": str(plan_path), "files": done}, indent=1))
    shutil.rmtree(tmp)


# --------------------------------------------------------------------------- check

def cmd_check(args) -> None:
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    out_dir = Path(plan.get("out_dir") or plan_path.parent / "exports").resolve()
    rendered = out_dir / "rendered.json"
    files = json.loads(rendered.read_text())["files"] if rendered.exists() else []
    problems, warns = [], []
    clips = plan["clips"]
    if len(files) != len(clips):
        problems.append(f"{len(clips)} clips in plan but {len(files)} rendered")
    for clip, f in zip(clips, files):
        tag = f"clip {clip['id']}"
        if not Path(f).exists():
            problems.append(f"{tag}: file missing {f}")
            continue
        m = probe(Path(f))
        if (m["width"], m["height"]) != (OUT_W, OUT_H):
            problems.append(f"{tag}: {m['width']}x{m['height']} instead of {OUT_W}x{OUT_H}")
        if not m["has_audio"]:
            problems.append(f"{tag}: no audio")
        want = float(clip["end"]) - float(clip["start"])
        if abs(m["duration"] - want) > 0.5:
            problems.append(f"{tag}: lasts {m['duration']:.1f}s, plan says {want:.1f}s")
        if m["duration"] > 60:
            problems.append(f"{tag}: {m['duration']:.0f}s is over the 60s cap")
        elif m["duration"] < 9:
            warns.append(f"{tag}: {m['duration']:.0f}s is very short (9-35s is the usual range)")
        hook = (clip.get("hook") or "").strip()
        low = hook.lower()
        if not hook:
            warns.append(f"{tag}: no hook (fine for about 1 clip in 3, as an A/B test)")
        else:
            if any(v in low for v in VAGUE_HOOKS):
                problems.append(f'{tag}: vague hook "{hook}" (vague hooks measured at 0.67x)')
            if len(hook) > 48:
                warns.append(f"{tag}: hook is {len(hook)} characters, hard to read in 2 seconds")
            if hook.isupper() and len(hook) > 12:
                warns.append(f"{tag}: all-caps hook (measured at 0.90x), prefer normal capitals")
        sheet = out_dir / f"qc_{int(clip['id']):02d}.jpg"
        ts = [0.6, m["duration"] / 2, max(0.6, m["duration"] - 0.6)]
        sel = "+".join(f"eq(n,{int(t * m['fps'])})" for t in ts)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f, "-vf", f"select='{sel}',scale=360:-2,tile=3x1",
                        "-frames:v", "1", "-vsync", "vfr", str(sheet)], capture_output=True)
        print(f"{tag}: {m['duration']:.1f}s ok -> look at {sheet}")
    for i, a in enumerate(clips):          # no footage reused between clips (3 s tolerance)
        for b in clips[i + 1:]:
            over = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
            if over > 3:
                problems.append(f"clips {a['id']} and {b['id']} share {over:.0f}s of footage")
    for w in warns:
        print("warning:", w)
    for p in problems:
        print("PROBLEM:", p)
    print("check passed" if not problems else f"check FAILED ({len(problems)} problems)")
    sys.exit(1 if problems else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    a = sub.add_parser("analyze")
    a.add_argument("video")
    a.add_argument("--work", help="work folder (default work/<video name>)")
    a.add_argument("--target", type=float, default=22, help="target clip length in seconds")
    a.add_argument("--max", type=int, default=0, help="number of candidates (default: 1 per 3 min, 3-25)")
    a.add_argument("--no-captions", action="store_true", help="skip transcription")
    a.add_argument("--no-motion", action="store_true", help="skip the picture-change signal")
    a.set_defaults(fn=cmd_analyze)
    f = sub.add_parser("frame")
    f.add_argument("video")
    f.add_argument("time", type=float)
    f.add_argument("--out")
    f.set_defaults(fn=cmd_frame)
    r = sub.add_parser("render")
    r.add_argument("plan")
    r.set_defaults(fn=cmd_render)
    c = sub.add_parser("check")
    c.add_argument("plan")
    c.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
