#!/usr/bin/env python3
"""TaleNoir channel art: profile logo (800x800) + banner (2560x1440, YouTube safe areas respected)."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from make_short import ROOT, font  # noqa: E402

OUT = ROOT / "youtube" / "channel"
SERIF = ("georgiab.ttf", "timesbd.ttf", "arialbd.ttf")
SANS = ("segoeuib.ttf", "arialbd.ttf")
SANS_R = ("segoeui.ttf", "arial.ttf")
AMBER, AMBER_HOT, INK = (255, 176, 32), (255, 214, 110), (6, 7, 10)
SS = 2


def radial(w, h, inner, outer, cx=.5, cy=.5, power=1.4):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx / w - cx) * (w / max(w, h))) ** 2 + ((yy / h - cy) * (h / max(w, h))) ** 2) / .72
    t = np.clip(r, 0, 1)[..., None] ** power
    arr = np.array(inner, np.float32) * (1 - t) + np.array(outer, np.float32) * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def vgrad(w, h, top, bottom):
    t = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    return Image.fromarray(np.repeat(arr, w, 1).astype(np.uint8), "RGB")


def keyhole_mask(size, cx, cy, s):
    """Keyhole silhouette (L mask). s = overall height."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    r = s * .25
    top = cy - s / 2 + r
    d.ellipse([cx - r, top - r, cx + r, top + r], fill=255)
    d.polygon([(cx - r * .42, top + r * .35), (cx + r * .42, top + r * .35),
               (cx + r * .95, cy + s / 2), (cx - r * .95, cy + s / 2)], fill=255)
    return m


def figure_mask(size, cx, base_y, h):
    """Noir silhouette (fedora + trench coat) standing in the light."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    top = base_y - h
    d.polygon([(cx - h * .17, top + h * .36), (cx + h * .17, top + h * .36), (cx + h * .25, top + h * .55),
               (cx + h * .3, base_y), (cx - h * .3, base_y), (cx - h * .25, top + h * .55)], fill=255)
    d.ellipse([cx - h * .24, top + h * .3, cx + h * .24, top + h * .46], fill=255)
    d.rectangle([cx - h * .05, top + h * .26, cx + h * .05, top + h * .36], fill=255)
    d.ellipse([cx - h * .1, top + h * .1, cx + h * .1, top + h * .32], fill=255)
    d.rounded_rectangle([cx - h * .12, top, cx + h * .12, top + h * .14], radius=h * .03, fill=255)
    d.ellipse([cx - h * .23, top + h * .1, cx + h * .23, top + h * .17], fill=255)
    return m


def light_fill(size, mask, cy_frac=.35):
    w, h = size
    g = radial(w, h, AMBER_HOT, (200, 80, 0), .5, cy_frac, 1.1).convert("RGBA")
    g.putalpha(mask)
    return g


def glow(mask, radius, color, strength=1.0):
    g = Image.new("RGBA", mask.size, color + (0,))
    g.putalpha(mask.filter(ImageFilter.GaussianBlur(radius)).point(lambda v: int(min(255, v * strength))))
    return g


def keyhole_emblem(px, ring=True):
    """Square RGBA emblem: dark disc, amber ring, glowing keyhole with a figure in the light."""
    S = px * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    disc = Image.new("L", (S, S), 0)
    ImageDraw.Draw(disc).ellipse([0, 0, S - 1, S - 1], fill=255)
    bg = radial(S, S, (30, 36, 52), (4, 5, 8), .5, .42, 1.2).convert("RGBA")
    bg.putalpha(disc)
    img.alpha_composite(bg)
    km = keyhole_mask((S, S), S / 2, S * .53, S * .6)
    img.alpha_composite(glow(km, S * .06, AMBER, 1.6))
    img.alpha_composite(light_fill((S, S), km))
    fm = figure_mask((S, S), S / 2, S * .53 + S * .3 + S * .01, S * .27)
    fm = ImageChops.multiply(fm, km)
    fig = Image.new("RGBA", (S, S), INK + (255,))
    fig.putalpha(fm)
    img.alpha_composite(fig)
    if ring:
        rd = ImageDraw.Draw(img)
        w = int(S * .022)
        inset = S * .045
        rd.ellipse([inset, inset, S - inset, S - inset], outline=AMBER + (255,), width=w)
    return img.resize((px, px), Image.Resampling.LANCZOS)


def logo():
    S = 800
    img = Image.new("RGB", (S, S), INK)
    em = keyhole_emblem(S)
    img.paste(em, (0, 0), em)
    return img


def rain(w, h, n, seed, alpha, length=(40, 110)):
    rng = np.random.default_rng(seed)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        x, y = rng.uniform(-200, w), rng.uniform(-200, h)
        L = rng.uniform(*length)
        d.line([(x, y), (x + L * .28, y + L)], fill=(190, 205, 230, int(alpha * rng.uniform(.4, 1))), width=2)
    return layer.filter(ImageFilter.GaussianBlur(.6))


def skyline(w, h, base, seed, color, hmin, hmax, win_p, win_col=AMBER):
    rng = np.random.default_rng(seed)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = -40
    while x < w:
        bw = rng.uniform(90, 260)
        bh = rng.uniform(hmin, hmax)
        top = base - bh
        d.rectangle([x, top, x + bw, h], fill=color)
        if rng.random() < .25:
            d.rectangle([x + bw * .4, top - 50, x + bw * .46, top], fill=color)
        for wy in np.arange(top + 22, base - 10, 34):
            for wx in np.arange(x + 14, x + bw - 18, 28):
                if rng.random() < win_p:
                    a = int(rng.uniform(120, 255))
                    d.rectangle([wx, wy, wx + 12, wy + 16], fill=win_col + (a,))
        x += bw + rng.uniform(0, 30)
    return layer


def train(w, y, h):
    layer = Image.new("RGBA", (w, y + h + 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([0, y + h + 20, w, y + h + 60], fill=(14, 16, 22, 255))
    for x in range(0, w, 160):
        d.rectangle([x, y + h + 60, x + 30, y + h + 200], fill=(14, 16, 22, 255))
    car = 520
    for cx in range(-60, w, car + 14):
        d.rounded_rectangle([cx, y, cx + car, y + h], 18, fill=(22, 26, 34, 255))
        for wx in range(cx + 40, cx + car - 60, 78):
            d.rounded_rectangle([wx, y + 28, wx + 52, y + h * .55], 8, fill=(255, 200, 110, 230))
    lights = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(lights)
    for cx in range(-60, w, car + 14):
        for wx in range(cx + 40, cx + car - 60, 78):
            ld.rectangle([wx - 10, y + 10, wx + 62, y + h * .6 + 20], fill=(255, 170, 60, 90))
    out = lights.filter(ImageFilter.GaussianBlur(26))
    out.alpha_composite(layer)
    return out


def banner():
    Wb, Hb = 2560, 1440
    img = vgrad(Wb, Hb, (8, 11, 20), (16, 20, 32)).convert("RGBA")
    moon = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(moon).ellipse([1880, 250, 2120, 490], fill=(255, 226, 170, 200))
    img.alpha_composite(moon.filter(ImageFilter.GaussianBlur(70)))
    img.alpha_composite(skyline(Wb, Hb, 980, 3, (20, 26, 40, 255), 220, 520, .05, (150, 170, 210)))
    fog = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(fog).rectangle([0, 820, Wb, 1060], fill=(80, 95, 125, 70))
    img.alpha_composite(fog.filter(ImageFilter.GaussianBlur(60)))
    img.alpha_composite(skyline(Wb, Hb, 1060, 9, (11, 14, 22, 255), 120, 420, .09))
    img.alpha_composite(train(Wb, 1040, 120))
    img.alpha_composite(rain(Wb, Hb, 900, 1, 46))

    # text panel shade over the safe area (center 1546x423 => x 507..2053, y 508..931)
    shade = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle([560, 520, 2000, 920], 80, fill=(4, 5, 8, 170))
    img.alpha_composite(shade.filter(ImageFilter.GaussianBlur(50)))

    em = keyhole_emblem(320)
    img.alpha_composite(em, (585, 560))
    d = ImageDraw.Draw(img)
    title_f = font(146, SERIF)
    x0 = 960
    title = "TALE NOIR"
    tx = x0
    for ch in title:  # wide letter spacing
        d.text((tx + 4, 720 + 6), ch, font=title_f, fill=(0, 0, 0, 200), anchor="ls")
        d.text((tx, 720), ch, font=title_f, fill=(245, 240, 228, 255), anchor="ls")
        tx += title_f.getlength(ch) + 14
    d.rectangle([x0 + 4, 752, x0 + 150, 758], fill=AMBER + (255,))
    d.text((x0 + 2, 812), "60-second mysteries. One twist you won't see coming.", font=font(42, SANS_R),
           fill=(232, 228, 220, 255), anchor="ls")
    d.text((x0 + 2, 880), "@TheTaleNoir", font=font(40, SANS), fill=AMBER + (255,), anchor="ls")
    return img.convert("RGB")


def guides(b):
    g = b.copy().convert("RGBA")
    d = ImageDraw.Draw(g)
    zones = [((0, 0, 2560, 1440), "TV (full)", (255, 80, 80)), ((0, 508, 2560, 931), "Desktop max width", (80, 200, 255)),
             ((507, 508, 2053, 931), "Safe area: all devices (1546x423)", (120, 255, 120))]
    for box, label, col in zones:
        d.rectangle(box, outline=col + (255,), width=6)
        d.text((box[0] + 16, box[1] + 12), label, font=font(34, SANS), fill=col + (255,))
    return g.convert("RGB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lg = logo()
    lg.save(OUT / "talenoir_logo_800.png")
    circ = Image.new("L", lg.size, 0)
    ImageDraw.Draw(circ).ellipse([0, 0, 799, 799], fill=255)
    preview = Image.new("RGB", (420, 220), (255, 255, 255))
    for i, sz in enumerate((176, 98, 40)):
        im = lg.resize((sz, sz), Image.Resampling.LANCZOS)
        preview.paste(im, (20 + i * 200 if i < 2 else 330, 22), circ.resize((sz, sz)))
    preview.save(OUT / "_logo_sizes_preview.png")
    b = banner()
    b.save(OUT / "talenoir_banner_2560x1440.jpg", quality=93)
    guides(b).resize((1280, 720)).save(OUT / "_banner_safe_area_preview.jpg", quality=88)
    print("channel art ->", OUT)


if __name__ == "__main__":
    main()
