#!/usr/bin/env python3
"""YouTube thumbnails for 'The Last Train at 2:17 AM' (vertical 1080x1920 + 1280x720)."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
import depth25 as d25  # noqa: E402
import make_short as ms  # noqa: E402
from make_short import BOLD, W, font  # noqa: E402

OUT = ms.ROOT / "youtube"


def punch(img):
    img = ImageEnhance.Contrast(img).enhance(1.18)
    return ImageEnhance.Color(img).enhance(1.25)


def text(d, xy, s, size, fill, anchor="mm", stroke=10):
    d.text(xy, s, font=font(size, BOLD), fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0))


def arrow(d, tip, tail, col=(255, 40, 40), w=26):
    import math
    ang = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
    L = w * 2.6
    base = (tip[0] - L * .8 * math.cos(ang), tip[1] - L * .8 * math.sin(ang))
    d.line([tail, base], fill=(0, 0, 0), width=w + 14)
    d.line([tail, base], fill=col, width=w)
    pts = [tip, (tip[0] - L * math.cos(ang - .5), tip[1] - L * math.sin(ang - .5)),
           (tip[0] - L * math.cos(ang + .5), tip[1] - L * math.sin(ang + .5))]
    d.polygon(pts, fill=col, outline=(0, 0, 0), width=6)


def red_glow(img, box, strength=150):
    """Red ring around the target, with a soft outer glow (face stays its own colour)."""
    w = max(8, int((box[2] - box[0]) * 0.035))
    g = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse(box, outline=(255, 30, 20, strength), width=w * 3)
    img.alpha_composite(g.filter(ImageFilter.GaussianBlur(18)))
    ring = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rd.ellipse(box, outline=(0, 0, 0, 255), width=w + 8)
    inset = [box[0] + 4, box[1] + 4, box[2] - 4, box[3] - 4]
    rd.ellipse(inset, outline=(255, 40, 30, 255), width=w)
    img.alpha_composite(ring)


def scene():
    cam = d25.Cam(dx=-30, zc=230, fx=500, fy=720, df=1600)
    img = d25.window_comp(1.0, cam, "scared", "calm", 1.0, 0.8, (440, 600)).convert("RGBA")
    return img, cam


def vertical(img, cam):
    v = img.copy()
    hx, hy = cam.pt(440, 600, 1900)
    red_glow(v, [hx - 200, hy - 200, hx + 200, hy + 200], 200)
    v = punch(v.convert("RGB")).convert("RGBA")
    v.alpha_composite(ms.A["vignette"])
    d = ImageDraw.Draw(v)
    shade = Image.new("RGBA", (W, 430), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle([0, 0, W, 430], fill=(0, 0, 0, 150))
    v.alpha_composite(shade.filter(ImageFilter.GaussianBlur(40)), (0, -60))
    text(d, (W / 2, 120), "WHO'S BEHIND", 104, (255, 255, 255))
    text(d, (W / 2, 250), "HER?", 150, (255, 210, 63), stroke=12)
    arrow(d, (hx + 175, hy + 120), (hx + 430, hy + 380))
    box = Image.new("RGBA", (W, 260), (0, 0, 0, 0))
    bd = ImageDraw.Draw(box)
    bd.rounded_rectangle([250, 40, 830, 220], 24, fill=(10, 10, 10, 235), outline=(70, 70, 70, 255), width=6)
    bd.text((540, 132), "2:17 AM", font=font(120, ms.MONO), fill=(255, 176, 32), anchor="mm")
    v.alpha_composite(box, (0, 1350))
    return v.convert("RGB")


def horizontal(img, cam):
    hw, hh = 1280, 720
    hz = Image.new("RGBA", (hw, hh), (8, 10, 16, 255))
    crop = img.crop((60, 180, 1020, 1350))
    s = hh / crop.height
    crop = crop.resize((int(crop.width * s), hh), Image.Resampling.LANCZOS)
    px = hw - crop.width + 40
    fade = Image.new("L", crop.size, 255)
    fd = ImageDraw.Draw(fade)
    for i in range(160):
        fd.line([(i, 0), (i, crop.height)], fill=int(255 * i / 160))
    crop.putalpha(fade)
    hz.alpha_composite(crop, (px, 0))
    hx, hy = cam.pt(440, 600, 1900)
    hx, hy = px + (hx - 60) * s, (hy - 180) * s
    red_glow(hz, [hx - 125, hy - 125, hx + 125, hy + 125], 200)
    hz = punch(hz.convert("RGB")).convert("RGBA")
    d = ImageDraw.Draw(hz)
    text(d, (40, 150), "WHO'S", 120, (255, 255, 255), "lm", 9)
    text(d, (40, 285), "BEHIND", 120, (255, 255, 255), "lm", 9)
    text(d, (40, 425), "HER?", 150, (255, 210, 63), "lm", 11)
    d.rounded_rectangle([40, 540, 420, 660], 18, fill=(10, 10, 10), outline=(70, 70, 70), width=5)
    d.text((230, 600), "2:17 AM", font=font(80, ms.MONO), fill=(255, 176, 32), anchor="mm")
    arrow(d, (hx - 108, hy + 75), (hx - 300, hy + 230), w=18)
    return hz.convert("RGB")


def main():
    OUT.mkdir(exist_ok=True)
    ms.build_assets()
    d25.build_depth_assets()
    img, cam = scene()
    vertical(img, cam).save(OUT / "thumbnail_vertical_1080x1920.jpg", quality=92)
    horizontal(img, cam).save(OUT / "thumbnail_1280x720.jpg", quality=92)
    print("thumbnails ->", OUT)


if __name__ == "__main__":
    main()
