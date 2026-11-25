# Dev-yt-shorts

Code-generated YouTube Shorts for the **TaleNoir** channel. Each short is drawn, voiced, scored and rendered by Python scripts. There is no hand-made footage. Every layer and audio stem is also written out as an editable file, so you can repaint, re-voice or re-cut the short in Krita, Audacity and Shotcut.

The current short is **The Last Train at 2:17 AM**: a 60-second, 1080×1920 vertical mystery. The script, shot list and art direction are in [story/the_last_train_217.md](story/the_last_train_217.md).

There are two cuts:

| Cut | Script | Output |
|---|---|---|
| Flat 2D cut-out | [build/make_short.py](build/make_short.py) | `output/the_last_train_217_24fps.mp4` |
| 2.5D depth / parallax | [build/depth25.py](build/depth25.py) | `output/the_last_train_217_2.5D_24fps.mp4` |

The 2.5D cut splits each shot into depth layers (background, midground, characters, blurred foreground, dust, fog) and moves a virtual pinhole camera through them. Near layers move faster than far ones, so the shots feel 3D while the art stays 2D.

## Requirements

- **Windows.** Narration uses the built-in Windows SAPI voice through [build/tts.ps1](build/tts.ps1), and fonts load from `C:\Windows\Fonts`.
- **Python 3** with `numpy` and `Pillow`:
  ```
  pip install numpy pillow
  ```
- **ffmpeg** on `PATH`.
- Optional: **Shotcut**. If its `melt` is found, the final edit renders from the generated `.mlt` project. If not, ffmpeg muxes picture and audio directly.
- Optional: **Krita**, to convert the layered art to native `.kra` files.

## Building

Run all commands from the repo root.

```
# 1. Flat cut: narration, audio stems, timing.json, art layers, final video
python build/make_short.py

# 2. 2.5D cut (needs the timing.json and stems from step 1)
python build/depth25.py

# Thumbnails -> youtube/
python build/thumbnail.py

# Channel logo + banner -> youtube/channel/
python build/channel_art.py

# Layered Krita projects -> assets/krita/
python build/krita_export.py
```

Both video builders accept `--stills`. It renders still frames plus a contact sheet for checking the shots, without the slow full render:

```
python build/make_short.py --stills   # -> output/stills/ + output/contact_sheet.png
python build/depth25.py --stills      # -> output/stills_2.5D/ + output/contact_sheet_2.5D.png
```

Every final render is loudness-normalised to −14 LUFS, the level YouTube targets.

## Project layout

```
story/            Script, narration, shot list and art direction for each short
build/
  make_short.py     Flat 2D builder: story timing, character and background drawing, TTS, SFX, music, render
  depth25.py        2.5D parallax builder; reuses make_short's timing, art and audio
  shotcut_project.py  Writes the Shotcut/MLT project and renders it with melt
  thumbnail.py      YouTube thumbnails (1080×1920 and 1280×720)
  channel_art.py    TaleNoir logo (800×800) and banner (2560×1440)
  krita_export.py   Packs assets/art/*.png into layered .kra projects
  tts.ps1           Windows SAPI narration renderer
assets/
  art/              Every generated layer (backgrounds, character poses, props)
  audio/            Narration lines, narration/SFX/music stems and the full mix
  krita/            Layered Krita projects with preview PNGs
output/           Rendered videos, stills, contact sheets, timing.json
youtube/          Thumbnails, channel art and upload_kit.md (title, description, tags)
*.mlt             Shotcut projects for each cut (V1 video plus narration, SFX and music tracks)
```

## Editing by hand

The pipeline produces a draft that you can finish in desktop tools:

- **Art:** open `assets/krita/*.kra` in Krita. Character expressions are separate layers, toggled on and off.
- **Voice:** the SAPI voice is a scratch track. Re-record the lines in `assets/audio/narration/` (the `*_raw.wav` files are unprocessed).
- **Edit:** open `the_last_train_217.mlt` or `the_last_train_217_2.5D.mlt` in Shotcut to rebalance the tracks or re-cut.

The end of the story doc covers this workflow step by step.

## Publishing

[youtube/upload_kit.md](youtube/upload_kit.md) has the title options, description, tags and hashtags, ready to paste into YouTube Studio.

## Adding a new short

1. Write the story doc in `story/`. Use the existing one as a template: script, scene timing, characters, backgrounds, SFX.
2. Copy `build/make_short.py` and edit the `SCENES` list (caption, TTS text and duration per scene) and the per-scene draw functions.
3. Build with `--stills` and check the contact sheet before the full render.
