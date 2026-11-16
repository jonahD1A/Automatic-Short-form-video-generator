---
name: gta6-clip
description: Turn a raw GTA 6 gameplay recording or stream VOD into finished 9:16 Shorts (TikTok, YouTube Shorts, Reels) - finds the best moments, writes the hook, reframes, burns in captions and exports. Use when the user drops a raw video and asks to clip it, edit it, "fais le montage", or make GTA 6 shorts.
---

# GTA 6 clipper

Raw video in, captioned vertical clips out. The script does the measuring and the rendering;
you make the three editorial calls: **which moment, which hook, which layout**.

All commands go through one script (`$S` = this skill's folder):

```bash
python3 $S/scripts/clipper.py setup                      # once: speech model, ~650 MB
python3 $S/scripts/clipper.py analyze <video>            # -> work/<name>/
python3 $S/scripts/clipper.py frame <video> <seconds>    # gridded frame, to locate a facecam
python3 $S/scripts/clipper.py render work/<name>/plan.json
python3 $S/scripts/clipper.py check  work/<name>/plan.json
```

Read `references/playbook.md` before choosing moments and writing hooks. It holds the moment
types, the measured hook table and the Rockstar rules.

## 1. Ask only what you cannot see

Start working as soon as you have the video path. Ask a question only if one of these is unknown
and matters: the language for hooks (default: the language spoken in the video, else the user's),
and whether the footage is the user's own or a streamer's (rights, see step 6).

## 2. Analyze

Run `setup` if `analyze` warns that the speech model is missing, then `analyze`. It writes into
`work/<video name>/`:

- `candidates.md` - ranked moments with times, the reason they scored, and what is said
- `sheets/cand_NN.jpg` - six frames per candidate
- `transcript.json` - every word with its time
- `scores.json` - the score of every second

Each second is scored from loudness above the video's own level (0.30), sudden rises (0.25),
hype words (0.20), picture change (0.15) and speech rate (0.10). That finds loud moments. It
cannot tell a funny one from a merely loud one: that is your job.

## 3. Pick the moments

Read `candidates.md` and **look at every sheet** with the Read tool. Keep a candidate only if a
viewer with no context understands it and something actually happens (see the moment types in
the playbook). Drop menus, loading screens, map browsing and dead driving.

For each one you keep, set exact times from `transcript.json`:

- **Start on the action or the first word of the setup.** No "so, um" lead-in. The first frame
  is the thumbnail.
- **End on the result**, 0.5-1 s after the payoff or the reaction. Never cut mid-word.
- Length: 9-35 s for a reaction or a moment (aim around 22 s), up to 58 s for an explanation.
  Never over 60 s.
- Two clips may not share more than 3 s of footage.

If the user asked for a number of clips, deliver that number only if that many moments are worth
it; otherwise deliver fewer and say why.

## 4. Choose layout

| layout | what it does | use when |
|---|---|---|
| `focus` (default) | 4:5 slice of the game over a blurred copy | gameplay with no facecam; the action is near the centre |
| `crop` | full-height 9:16 slice | the subject stays in the middle third; maximum impact |
| `fit` | whole 16:9 frame over a blurred copy | the HUD, minimap or both sides of the screen matter |
| `split` | facecam on top, gameplay below | a streamer's face is on screen |

`crop_x` (0 = left edge, 1 = right edge, default 0.5) moves the slice. For `split`, run `frame`
on a second where the facecam is visible, read the grid, and give
`"facecam": {"x": .., "y": .., "w": .., "h": ..}` as fractions of the frame.

## 5. Write the hook, then the plan

One hook per clip, on screen for the first 4 s. Follow the hook table in the playbook. The rules
that matter most:

- Name what the viewer is about to see, **without giving away the ending**.
- Under 45 characters. Normal capitals, not all caps.
- Never vague ("Wait for it", "Regarde ça"): `check` rejects those.
- The hook must be true. A promise the clip does not keep is clickbait.
- Post about one clip in three with no hook (`"hook": ""`), to compare.

Write `work/<name>/plan.json`:

```json
{
  "source": "/absolute/path/to/raw.mp4",
  "clips": [
    {
      "id": 1,
      "start": 754.2,
      "end": 776.8,
      "title": "cop car backflip",
      "hook": "He stole a cop car in 4 seconds",
      "hook_keyword": "4 seconds",
      "layout": "focus",
      "crop_x": 0.5
    }
  ]
}
```

Optional per clip: `hook_style` (`"box"` = black text on a white box), `hook_seconds`,
`captions` (`false` to turn off), `caption_y` / `hook_y` (0-1, vertical position),
`words` (hand-corrected captions: `[{"w": "word", "s": 0.4, "e": 0.7}]`, times from clip start).

Before rendering, read the transcript of each clip. If a name or a GTA term is misheard
(Vice City, Leonida, Lucia, Jason), fix it in `transcript.json` or pass `words`.

## 6. Render, check, look

Run `render`, then `check`. `check` verifies size (1080x1920), length, audio, footage reuse and
hook rules, and writes `exports/qc_NN.jpg` (start, middle, end of each clip).

**Open every `qc_NN.jpg`.** Fix and re-render if the subject is cut off, the hook or captions
cover the action or a face, a caption is wrong, or the first frame is dull. Do not report a clip
as done before you have looked at it.

## 7. Hand over

Give the user, per clip: the file path, the hook, a title and a caption with 3-5 hashtags
(`#gta6 #gtavi` plus the moment), and one line on why this moment. Then state plainly:

- anything you could not verify (misheard words you guessed, a moment you were unsure about);
- the rights reminder below, once.

## Rules that are not optional

- **Rights.** The user's own gameplay is fine to post. A streamer's footage needs that
  streamer's permission or a clipping campaign that grants it. If you do not know, ask.
- **Rockstar's policy.** No footage from leaked or early copies before the official release.
  No clip of the ending, no cutscene-only clip, no big story reveal. Gameplay moments are fine.
  If a candidate is a cutscene, skip it and say so.
- **Music.** In-game radio is licensed music and gets claimed. Prefer moments with the radio off;
  if a song is clearly audible, warn the user. Never add a music track the user did not supply.
- **No fake views, no misleading titles.** Campaigns verify views and cancel the payout.

## Limits to tell the user about when they matter

- Captions come from automatic transcription (25 European languages, English and French
  included). Names and slang can be wrong, and very loud game audio hides speech.
- Moments with no sound spike and no speech (a silent visual gag) are missed by the scoring.
  If the user knows a timestamp, put it in the plan directly.
- The crop is fixed per clip. It does not follow a subject that moves across the screen: use
  `fit`, or split the moment into two clips with different `crop_x`.
- No added music, sound effects or zooms.
