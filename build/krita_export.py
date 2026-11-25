#!/usr/bin/env python3
"""
Packs the generated art (assets/art/*.png) into layered Krita projects (assets/krita/*.kra).

Each project is first written as OpenRaster (.ora, layered), then converted by Krita's own
command line (krita --export) so you get native .kra files to repaint in Krita.
Run after build/make_short.py.
"""
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "assets" / "art"
KRA = ROOT / "assets" / "krita"

# project -> (canvas size, [(layer name, png, x, y, visible)])  -- listed bottom to top
PROJECTS = {
    "maya_character_sheet": ((1600, 2000), [
        ("body_seated", "maya_body_seated.png", 0, 0, True),
        ("head_tired", "maya_head_tired.png", 140, 30, True),
        ("head_uneasy", "maya_head_uneasy.png", 140, 30, False),
        ("head_scared", "maya_head_scared.png", 140, 30, False),
        ("head_shock", "maya_head_shock.png", 140, 30, False),
        ("head_dread", "maya_head_dread.png", 140, 30, False),
        ("head_turned_scared", "maya_head_turned_scared.png", 140, 30, False),
        ("standing_step0", "maya_standing_step0.png", 700, 100, True),
        ("standing_step1", "maya_standing_step1.png", 1150, 100, True),
    ]),
    "stranger_character_sheet": ((1300, 1050), [
        ("calm", "stranger_calm.png", 20, 20, True),
        ("urgent_with_phone", "stranger_urgent_phone.png", 660, 20, True),
    ]),
    "sc2_platform": ((1080, 1920), [
        ("bg_platform", "bg_platform.png", 0, 0, True),
        ("far_train_downtown", "prop_train_far_downtown.png", -400, 690, True),
        ("near_train_not_in_service", "prop_train_near_out_of_service.png", -530, 1000, True),
        ("maya_standing", "maya_standing_step0.png", 610, 953, True),
    ]),
    "sc1_4_5_7_window": ((1080, 1920), [
        ("bg_window", "bg_window.png", 0, 0, True),
        ("other_train_window", "prop_other_train_window.png", 60, 400, True),
        ("stranger", "stranger_calm.png", 234, 397, True),
    ]),
    "sc3_4_8_interior": ((1080, 1920), [
        ("bg_interior", "bg_interior.png", 0, 0, True),
        ("maya_body", "maya_body_seated.png", 380, 550, True),
        ("maya_head", "maya_head_tired.png", 520, 580, True),
    ]),
    "sc6_phone": ((1080, 1920), [("phone_wrong_train", "prop_phone_wrong_train.png", 0, 0, True)]),
    "sc6_door_sign_tilt": ((1080, 3000), [("bg_door_sign_tilt", "bg_door_sign_tilt.png", 0, 0, True)]),
    "sc7_aerial": ((1080, 1920), [("bg_aerial_tracks", "bg_aerial_tracks.png", 0, 0, True)]),
    "end_card": ((1080, 1920), [("end_card", "end_card.png", 0, 0, True)]),
}


def find_krita():
    cands = [shutil.which("krita")]
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "") + r"\Programs"):
        cands += [str(Path(base) / "Krita (x64)" / "bin" / "krita.exe"), str(Path(base) / "Krita" / "bin" / "krita.exe")]
    return next((c for c in cands if c and Path(c).exists()), None)


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def write_ora(path, size, layers):
    w, h = size
    merged = Image.new("RGBA", size, (0, 0, 0, 0))
    stack = []
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "image/openraster", compress_type=zipfile.ZIP_STORED)
        for i, (name, png, x, y, vis) in enumerate(layers):
            img = Image.open(ART / png).convert("RGBA")
            z.writestr(f"data/layer{i}.png", png_bytes(img), compress_type=zipfile.ZIP_DEFLATED)
            stack.insert(0, f'  <layer name={quoteattr(name)} src="data/layer{i}.png" x="{x}" y="{y}" '
                            f'opacity="1.0" visibility="{"visible" if vis else "hidden"}"/>')
            if vis:
                tmp = Image.new("RGBA", size, (0, 0, 0, 0))
                tmp.paste(img, (x, y), img)
                merged = Image.alpha_composite(merged, tmp)
        xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<image version="0.0.3" w="{w}" h="{h}">\n<stack>\n'
               + "\n".join(stack) + "\n</stack>\n</image>\n")
        z.writestr("stack.xml", xml, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("mergedimage.png", png_bytes(merged), compress_type=zipfile.ZIP_DEFLATED)
        thumb = merged.copy()
        thumb.thumbnail((256, 256))
        z.writestr("Thumbnails/thumbnail.png", png_bytes(thumb), compress_type=zipfile.ZIP_DEFLATED)


def main():
    KRA.mkdir(parents=True, exist_ok=True)
    krita = find_krita()
    print("Krita:", krita or "NOT FOUND (leaving .ora files, which Krita also opens with layers)")
    for name, (size, layers) in PROJECTS.items():
        ora = KRA / f"{name}.ora"
        write_ora(ora, size, layers)
        if not krita:
            continue
        kra = KRA / f"{name}.kra"
        preview = KRA / f"{name}_preview.png"
        for target in (kra, preview):
            r = subprocess.run([krita, str(ora), "--export", "--export-filename", str(target)],
                               capture_output=True, text=True, timeout=300)
            ok = target.exists()
            print(f"  {target.name}: {'ok' if ok else 'FAILED'}" + ("" if ok else f" ({r.stderr.strip()[:200]})"))
        if kra.exists():
            ora.unlink()


if __name__ == "__main__":
    sys.exit(main())
