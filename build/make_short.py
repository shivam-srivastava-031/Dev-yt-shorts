#!/usr/bin/env python3
"""
The Last Train at 2:17 AM -- YouTube Short builder.

Draws every layer (backgrounds + character poses), voices the narration with the
Windows SAPI voice (scratch track), synthesises all sound effects and music, and
renders a 1080x1920 / 30 fps MP4 with ffmpeg.

All layers and audio stems are also written to assets/ so they can be repainted in
Krita, rigged in OpenToonz, re-voiced in Audacity and re-cut in Shotcut.

Usage:
    python build/make_short.py            # full build -> output/the_last_train_217.mp4
    python build/make_short.py --stills   # QA stills only -> output/stills/
"""
import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
ART_DIR = ROOT / "assets" / "art"
AUDIO_DIR = ROOT / "assets" / "audio"
NARR_DIR = AUDIO_DIR / "narration"

W, H, FPS, SR, SS = 1080, 1920, 24, 44100, 2
LZ = Image.Resampling.LANCZOS
BC = Image.Resampling.BICUBIC
FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
BOLD = ("ariblk.ttf", "impact.ttf", "arialbd.ttf")
TALL = ("impact.ttf", "ariblk.ttf", "arialbd.ttf")
MONO = ("consolab.ttf", "consola.ttf", "courbd.ttf", "cour.ttf")
SANS = ("segoeuib.ttf", "arialbd.ttf")


# --------------------------------------------------------------------------- story
def L(cap, tts=None, pre=0.0, hl=False):
    return dict(cap=cap, tts=tts or cap, pre=pre, hl=hl)


SCENES = [
    ("s1", 5.5, [L("The man behind Maya didn't exist."),
                 L("She could only see him in the window.")]),
    ("s2", 8.5, [L("2:17 AM.", "Two seventeen A.M."),
                 L("Two trains pulled in, side by side."),
                 L("Exhausted, Maya jumped on the closest one.")]),
    ("s3", 5.8, [L("Her car was empty."),
                 L("Just the hum of the rails, and her tired reflection.")]),
    ("s4", 7.8, [L("Then she saw him."),
                 L("A man in a yellow raincoat, right behind her."),
                 L("She turned."),
                 L("Empty seat.")]),
    ("s5", 7.0, [L("The lights flickered."),
                 L("In the glass, he was closer, pressing his phone to the window.")]),
    ("s6", 7.5, [L("Two words."),
                 L("WRONG TRAIN.", "Wrong train.", pre=0.15, hl=True),
                 L("Above the door, the sign read:"),
                 L("NOT IN SERVICE.", "Not in service.", pre=0.15, hl=True)]),
    ("s7", 6.5, [L("The tracks split."),
                 L("He was never behind her."),
                 L("He was on the other train.")]),
    ("s8", 9.0, [L("Hers rolled into the dark yard."),
                 L("Doors locked.", pre=0.1),
                 L("The last train at 2:17 left on time.", "The last train at two seventeen left on time."),
                 L("Maya just wasn't on it.", pre=0.55, hl=True)]),
]
MAX_TOTAL = 59.9
SLOW = 0.97      # narration playback speed (<1 = slower, deeper)


# --------------------------------------------------------------------------- helpers
def C(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


@functools.lru_cache(None)
def font(size, names):
    for n in names + ("arialbd.ttf", "arial.ttf"):
        p = FONTS / n
        if p.exists():
            return ImageFont.truetype(str(p), int(size))
    return ImageFont.load_default(int(size))


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def lerp(a, b, x):
    return a + (b - a) * x


class Cv:
    """Supersampled drawing canvas in logical units."""

    def __init__(self, w, h, k=1.0, bg=(0, 0, 0, 0)):
        self.w, self.h, self.k = w, h, k
        self.f = k * SS
        self.im = Image.new("RGBA", (max(1, round(w * self.f)), max(1, round(h * self.f))), bg)
        self.d = ImageDraw.Draw(self.im)

    def _b(self, x0, y0, x1, y1):
        return [x0 * self.f, y0 * self.f, x1 * self.f, y1 * self.f]

    def _p(self, pts):
        return [(x * self.f, y * self.f) for x, y in pts]

    def _w(self, w):
        return max(1, round(w * self.f))

    def ell(self, cx, cy, rx, ry, fill=None, outline=None, width=0):
        self.d.ellipse(self._b(cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline,
                       width=self._w(width) if outline else 0)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=0, r=0):
        b = self._b(x0, y0, x1, y1)
        wd = self._w(width) if outline else 0
        if r:
            rr = min(r * self.f, (b[2] - b[0]) / 2, (b[3] - b[1]) / 2)
            self.d.rounded_rectangle(b, radius=rr, fill=fill, outline=outline, width=wd)
        else:
            self.d.rectangle(b, fill=fill, outline=outline, width=wd)

    def poly(self, pts, fill=None, outline=None, width=0):
        self.d.polygon(self._p(pts), fill=fill, outline=outline, width=self._w(width) if outline else 0)

    def line(self, pts, fill, width):
        self.d.line(self._p(pts), fill=fill, width=self._w(width), joint="curve")

    def cap(self, pts, fill, width):
        self.line(pts, fill, width)
        for x, y in (pts[0], pts[-1]):
            self.ell(x, y, width / 2, width / 2, fill=fill)

    def arc(self, cx, cy, rx, ry, a0, a1, fill, width):
        self.d.arc(self._b(cx - rx, cy - ry, cx + rx, cy + ry), a0, a1, fill=fill, width=self._w(width))

    def chord(self, cx, cy, rx, ry, a0, a1, fill):
        self.d.chord(self._b(cx - rx, cy - ry, cx + rx, cy + ry), a0, a1, fill=fill)

    def text(self, x, y, s, names, size, fill, anchor="mm", stroke=0, stroke_fill=None):
        self.d.text((x * self.f, y * self.f), s, font=font(round(size * self.f), names), fill=fill,
                    anchor=anchor, stroke_width=round(stroke * self.f), stroke_fill=stroke_fill)

    def out(self):
        return self.im.resize((max(1, round(self.w * self.k)), max(1, round(self.h * self.k))), LZ)


def comp(base, img, x, y, alpha=1.0):
    """alpha_composite that tolerates negative / out-of-bounds positions."""
    x, y = int(round(x)), int(round(y))
    if alpha < 0.999:
        if alpha <= 0.001:
            return
        img = with_alpha(img, alpha)
    sx, sy = max(0, -x), max(0, -y)
    dx, dy = max(0, x), max(0, y)
    w = min(img.width - sx, base.width - dx)
    h = min(img.height - sy, base.height - dy)
    if w <= 0 or h <= 0:
        return
    base.alpha_composite(img, (dx, dy), (sx, sy, sx + w, sy + h))


def with_alpha(img, a):
    r, g, b, al = img.split()
    al = al.point(lambda v, a=a: int(v * a))
    return Image.merge("RGBA", (r, g, b, al))


def tint(img, mul):
    r, g, b, a = img.split()
    r = r.point(lambda v: int(v * mul[0]))
    g = g.point(lambda v: int(v * mul[1]))
    b = b.point(lambda v: int(v * mul[2]))
    return Image.merge("RGBA", (r, g, b, a))


def clip_to(img, mask):
    img.putalpha(ImageChops.multiply(img.getchannel("A"), mask))
    return img


def scaled(img, s):
    if abs(s - 1) < 1e-3:
        return img
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), BC)


def camera(img, z=1.0, cx=W / 2, cy=H / 2):
    if abs(z - 1) < 1e-4 and abs(cx - W / 2) < 0.5 and abs(cy - H / 2) < 0.5:
        return img
    w, h = W / z, H / z
    x0 = min(max(cx - w / 2, 0), W - w)
    y0 = min(max(cy - h / 2, 0), H - h)
    return img.resize((W, H), BC, box=(x0, y0, x0 + w, y0 + h))


def darken(img, level, tint_rgb=(20, 30, 60), tint_amt=0.25):
    """Lights level 0..1 -> darker, cooler image (RGB)."""
    if level >= 0.999:
        return img
    out = ImageEnhance.Brightness(img).enhance(max(level, 0.0))
    solid = Image.new("RGB", img.size, tint_rgb)
    return Image.blend(out, solid, tint_amt * (1 - level) * 0.6)


def mask_of(draw_fn):
    c = Cv(W, H)
    draw_fn(c)
    return c.out().getchannel("A")


# --------------------------------------------------------------------------- palette
INK = C("#1b1414")
SKIN_M, SKIN_M_SH, HAIR_M = C("#8d5a3b"), C("#6c4129"), C("#1c1311")
JACKET, JACKET_SH = C("#5d616c"), C("#474a53")
SCRUB, SCRUB_SH = C("#2a9d8f"), C("#1f776c")
SHOE, TOTE = C("#eceae4"), C("#c99a3b")
SKIN_S, SKIN_S_SH, HAIR_S = C("#e0b08c"), C("#b9845f"), C("#3b2a1e")
COAT, COAT_SH = C("#e9b824"), C("#b8890f")

EXPR = {
    "tired":  dict(open=.42, brow=-.30, tilt=-.04, mouth="flat", bags=True, pupil=.075),
    "uneasy": dict(open=.80, brow=-.36, tilt=.07, mouth="down", pupil=.075),
    "scared": dict(open=1.0, brow=-.44, tilt=.13, mouth="o", pupil=.06),
    "shock":  dict(open=1.12, brow=-.50, tilt=.12, mouth="O", pupil=.05),
    "dread":  dict(open=.85, brow=-.40, tilt=.15, mouth="wavy", pupil=.065),
    "calm":   dict(open=.62, brow=-.32, tilt=-.02, mouth="flat", pupil=.08),
    "urgent": dict(open=.98, brow=-.44, tilt=.12, mouth="open", pupil=.065),
}


# --------------------------------------------------------------------------- characters
def draw_face(c, cx, cy, r, expr, gaze, skin_sh, hair, turn=0.0):
    e = EXPR[expr]
    fx = cx + turn * 0.22 * r
    ey = cy + 0.05 * r
    erx, ery = 0.19 * r, 0.15 * r * e["open"]
    for side in (-1, 1):
        ex = fx + side * 0.40 * r
        c.ell(ex, ey, erx, ery, fill=C("#fbf7ef"), outline=INK, width=0.025 * r)
        pr = min(e["pupil"] * r, ery * 0.92)
        px = ex + gaze[0] * (erx - pr) * 0.85
        py = ey + gaze[1] * max(ery - pr, 0) * 0.8
        c.ell(px, py, pr, pr, fill=C("#22140f"))
        c.ell(px - pr * .35, py - pr * .35, pr * .28, pr * .28, fill=C("#ffffff"))
        if e["open"] < .5:
            c.arc(ex, ey, erx * 1.05, ery * 1.35, 190, 350, INK, 0.045 * r)
        by = cy + e["brow"] * r
        inner = (ex - side * 0.16 * r, by - e["tilt"] * r)
        outer = (ex + side * 0.22 * r, by + e["tilt"] * 0.5 * r)
        c.cap([inner, outer], hair, 0.06 * r)
        if e.get("bags"):
            c.arc(ex, ey + ery + 0.06 * r, erx * 0.85, 0.06 * r, 20, 160, skin_sh, 0.025 * r)
    c.arc(fx + turn * 0.1 * r, cy + 0.30 * r, 0.08 * r, 0.06 * r, 20, 160, skin_sh, 0.035 * r)
    my, m, mc = cy + 0.58 * r, e["mouth"], C("#4a2219")
    if m == "flat":
        c.cap([(fx - .14 * r, my), (fx + .14 * r, my)], mc, .035 * r)
    elif m == "down":
        c.arc(fx, my + .07 * r, .14 * r, .08 * r, 200, 340, mc, .035 * r)
    elif m == "o":
        c.ell(fx, my, .07 * r, .09 * r, fill=C("#3a1410"), outline=mc, width=.02 * r)
    elif m == "O":
        c.ell(fx, my + .03 * r, .11 * r, .15 * r, fill=C("#3a1410"), outline=mc, width=.02 * r)
    elif m == "wavy":
        pts = [(fx - .15 * r + i * .05 * r, my + (.016 * r if i % 2 else -.016 * r)) for i in range(7)]
        c.line(pts, mc, .03 * r)
    elif m == "open":
        c.ell(fx, my, .12 * r, .07 * r, fill=C("#3a1410"), outline=mc, width=.02 * r)


def draw_maya_head(c, cx, cy, r, expr, gaze=(0, 0), turn=0.0):
    w = 0.03 * r
    c.ell(cx - turn * .2 * r, cy - 1.12 * r, .48 * r, .44 * r, fill=HAIR_M, outline=INK, width=w)
    c.ell(cx, cy - .1 * r, 1.08 * r, 1.12 * r, fill=HAIR_M, outline=INK, width=w)
    for side in (-1, 1):
        if side * turn > .5:
            continue
        c.ell(cx + side * .98 * r + turn * .1 * r, cy + .14 * r, .16 * r, .22 * r, fill=SKIN_M, outline=INK, width=w)
        c.ell(cx + side * .98 * r + turn * .1 * r, cy + .36 * r, .04 * r, .04 * r, fill=C("#e3b341"))
    c.ell(cx + turn * .06 * r, cy + .08 * r, r, 1.1 * r, fill=SKIN_M, outline=INK, width=w)
    c.chord(cx + turn * .05 * r, cy - .14 * r, 1.05 * r, 1.02 * r, 198, 342, fill=HAIR_M)
    c.arc(cx + turn * .05 * r, cy - .2 * r, .7 * r, .62 * r, 225, 300, C("#3a2a26"), .05 * r)
    c.ell(cx - turn * .2 * r, cy - .8 * r, .22 * r, .08 * r, fill=C("#c0392b"))
    draw_face(c, cx, cy + .08 * r, r, expr, gaze, SKIN_M_SH, HAIR_M, turn)


def draw_maya_seated_body(c):
    c.rect(185, 760, 285, 970, fill=SCRUB_SH, outline=INK, width=3, r=20)
    c.rect(315, 760, 415, 970, fill=SCRUB_SH, outline=INK, width=3, r=20)
    c.rect(170, 945, 295, 995, fill=SHOE, outline=INK, width=3, r=18)
    c.rect(305, 945, 430, 995, fill=SHOE, outline=INK, width=3, r=18)
    c.rect(165, 670, 435, 790, fill=SCRUB, outline=INK, width=3, r=40)
    c.rect(268, 290, 332, 380, fill=SKIN_M, outline=INK, width=3)
    c.rect(150, 360, 450, 730, fill=JACKET, outline=INK, width=4, r=80)
    c.poly([(240, 362), (360, 362), (335, 724), (265, 724)], fill=SCRUB)
    c.poly([(266, 362), (334, 362), (300, 432)], fill=SKIN_M)
    c.line([(240, 365), (264, 725)], JACKET_SH, 6)
    c.line([(360, 365), (336, 725)], JACKET_SH, 6)
    c.line([(276, 380), (316, 520)], C("#3557a6"), 5)
    c.line([(326, 380), (324, 520)], C("#3557a6"), 5)
    c.rect(296, 512, 352, 582, fill=C("#f4f4f4"), outline=INK, width=2, r=6)
    c.rect(296, 512, 352, 528, fill=C("#3557a6"), r=4)
    c.rect(304, 536, 322, 558, fill=C("#9aa0a8"))
    for side in (-1, 1):
        pts = [(300 + side * 125, 405), (300 + side * 160, 565), (300 + side * 85, 690)]
        c.cap(pts, INK, 76)
        c.cap(pts, JACKET, 68)
        c.ell(300 + side * 68, 692, 36, 30, fill=SKIN_M, outline=INK, width=3)


def draw_maya_standing(c, step=0):
    a, b = (830, 812) if step == 0 else (812, 830)
    c.rect(148, 520, 198, a, fill=SCRUB_SH, outline=INK, width=3, r=16)
    c.rect(202, 520, 252, b, fill=SCRUB_SH, outline=INK, width=3, r=16)
    c.rect(134, a - 12, 204, a + 22, fill=SHOE, outline=INK, width=3, r=12)
    c.rect(196, b - 12, 266, b + 22, fill=SHOE, outline=INK, width=3, r=12)
    c.rect(180, 175, 220, 245, fill=SKIN_M, outline=INK, width=3)
    c.rect(126, 230, 274, 545, fill=JACKET, outline=INK, width=4, r=44)
    c.poly([(180, 232), (220, 232), (212, 540), (188, 540)], fill=SCRUB)
    c.poly([(184, 232), (216, 232), (200, 270)], fill=SKIN_M)
    c.cap([(140, 255), (116, 400), (126, 505)], INK, 44)
    c.cap([(140, 255), (116, 400), (126, 505)], JACKET, 38)
    c.ell(126, 512, 20, 18, fill=SKIN_M, outline=INK, width=3)
    c.line([(250, 245), (296, 430)], C("#7a5a1c"), 7)
    c.rect(258, 418, 352, 566, fill=TOTE, outline=INK, width=3, r=12)
    c.cap([(260, 255), (284, 390), (280, 470)], INK, 44)
    c.cap([(260, 255), (284, 390), (280, 470)], JACKET, 38)
    c.ell(280, 476, 20, 18, fill=SKIN_M, outline=INK, width=3)
    draw_maya_head(c, 200, 120, 58, "tired", (0, 0.3))


def draw_stranger(c, expr, phone=False):
    c.rect(70, 380, 530, 1000, fill=COAT, outline=INK, width=4, r=150)
    c.line([(300, 430), (300, 960)], COAT_SH, 6)
    for y in (540, 660, 780):
        c.rect(286, y - 14, 314, y + 14, fill=C("#5a3b1c"), r=6)
    c.ell(300, 250, 150, 170, fill=COAT, outline=INK, width=4)
    c.ell(300, 268, 112, 132, fill=C("#6b4f0c"))
    c.ell(300, 282, 86, 100, fill=SKIN_S, outline=INK, width=3)
    c.chord(300, 238, 92, 72, 188, 352, fill=HAIR_S)
    for x, dx in ((252, -4), (348, 4)):
        c.line([(x, 398), (x + dx, 480)], C("#f2efe6"), 5)
        c.ell(x + dx, 486, 7, 7, fill=C("#f2efe6"))
    draw_face(c, 300, 282, 86, expr, (0, 0), SKIN_S_SH, HAIR_S)
    if phone:
        pts = [(480, 1000), (440, 780), (352, 652)]
        c.cap(pts, INK, 104)
        c.cap(pts, COAT, 94)
        c.rect(238, 440, 372, 664, fill=C("#111214"), outline=C("#4a4d55"), width=3, r=16)
        c.rect(250, 456, 360, 648, fill=C("#eef5ff"), r=8)
        c.rect(266, 520, 344, 546, fill=C("#1a1a1a"), r=4)
        c.rect(266, 556, 344, 582, fill=C("#1a1a1a"), r=4)
        c.ell(338, 648, 54, 44, fill=SKIN_S, outline=INK, width=3)
        c.ell(374, 584, 16, 34, fill=SKIN_S, outline=INK, width=3)
        c.ell(236, 600, 16, 26, fill=SKIN_S, outline=INK, width=3)


@functools.lru_cache(None)
def maya_body(k=1.0):
    c = Cv(600, 1000, k)
    draw_maya_seated_body(c)
    return c.out()


@functools.lru_cache(None)
def maya_head(expr, gaze=(0, 0), turn=0.0, k=1.0):
    c = Cv(320, 360, k)
    draw_maya_head(c, 160, 200, 92, expr, gaze, turn)
    return c.out()


def maya_seated(expr, gaze=(0, 0), turn=0.0, tilt=0.0, k=1.0):
    body = maya_body(k).copy()
    head = maya_head(expr, gaze, turn, k)
    if abs(tilt) > 0.05:
        head = head.rotate(tilt, resample=BC, center=(160 * k, 300 * k))
    comp(body, head, 140 * k, 30 * k)
    return body


@functools.lru_cache(None)
def maya_reflection(expr, gaze=(0, 0)):
    img = ImageOps.mirror(maya_seated(expr, gaze))
    img = tint(img, (.55, .72, .95))
    return with_alpha(img, 0.58)


@functools.lru_cache(None)
def maya_standing(step=0, k=0.95):
    c = Cv(400, 880, k)
    draw_maya_standing(c, step)
    return c.out()


@functools.lru_cache(None)
def stranger(expr, phone=False):
    c = Cv(600, 1000)
    draw_stranger(c, expr, phone)
    return c.out()


# --------------------------------------------------------------------------- backgrounds
def glow_layer(size, draw_fn, radius):
    g = Image.new("RGB", size, (0, 0, 0))
    draw_fn(ImageDraw.Draw(g))
    return g.filter(ImageFilter.GaussianBlur(radius))


def bg_platform():
    c = Cv(W, H)
    c.rect(0, 0, W, 252, fill=C("#121418"))
    for x in (60, 420, 780):
        c.rect(x, 60, x + 240, 86, fill=C("#2a2d31"), r=6)
        c.rect(x + 8, 86, x + 232, 98, fill=C("#fff3c4"), r=4)
    c.rect(0, 252, W, 700, fill=C("#a3966f"))
    for row, y in enumerate(range(252, 700, 38)):
        c.line([(0, y), (W, y)], C("#857a59"), 2)
        off = 0 if row % 2 == 0 else 38
        for x in range(-38 + off, W + 76, 76):
            c.line([(x, y), (x, y + 38)], C("#857a59"), 2)
    c.rect(0, 430, W, 520, fill=C("#1d3557"))
    c.rect(0, 430, W, 438, fill=C("#e3e3e3"))
    c.rect(0, 512, W, 520, fill=C("#e3e3e3"))
    c.text(540, 476, "CORLISS AVE", BOLD, 58, C("#f1f1f1"))
    c.rect(0, 650, W, 700, fill=C("#5f5644"))
    c.rect(0, 700, W, 1000, fill=C("#0c0d10"))
    for y in (968, 986):
        c.line([(0, y), (W, y)], C("#5a5e63"), 5)
    c.rect(0, 1000, W, 1440, fill=C("#08090b"))
    for x in range(0, W, 70):
        c.rect(x, 1404, x + 40, 1436, fill=C("#1a1715"))
    for y in (1398, 1424):
        c.line([(0, y), (W, y)], C("#6a6f75"), 6)
    c.rect(0, 1440, W, H, fill=C("#55585c"))
    c.rect(0, 1440, W, 1462, fill=C("#cfd2d4"))
    c.rect(0, 1462, W, 1508, fill=C("#e3b505"))
    for x in range(12, W, 26):
        for y in (1474, 1494):
            c.ell(x, y, 5, 5, fill=C("#b88f00"))
    for y in (1640, 1820):
        c.line([(0, y), (W, y)], C("#45484c"), 3)
    for x in (260, 620, 960):
        c.line([(x, 1508), (x - 40, H)], C("#45484c"), 3)
    c.ell(250, 1700, 150, 26, fill=C("#6d7a86"))
    c.ell(860, 1860, 190, 30, fill=C("#6a7783"))
    for x in (470, 610):
        c.rect(x - 6, 0, x + 6, 270, fill=C("#2b2e33"))
    c.rect(400, 262, 680, 382, fill=C("#0b0b0b"), outline=C("#3a3d42"), width=6, r=12)
    c.text(522, 322, "2:17", MONO, 92, C("#ffb020"))
    c.text(640, 346, "AM", BOLD, 26, C("#ffb020"))
    c.rect(440, 388, 640, 424, fill=C("#20252b"), r=6)
    c.text(540, 406, "LAST TRAIN", BOLD, 22, C("#f1f1f1"))
    img = c.out().convert("RGB")

    def g(d):
        for x in (60, 420, 780):
            d.rectangle([x + 8, 86, x + 232, 98], fill=(255, 220, 140))
        d.rectangle([420, 285, 620, 360], fill=(150, 80, 0))
    return ImageChops.add(img, glow_layer((W, H), g, 18))


def train_side(length, height, lit, sign, sign_col, doors, yellow_at=None):
    c = Cv(length, height)
    body, body_sh = C("#b8bdc3"), C("#959ca5")
    c.rect(0, 0, length + 80, height - 28, fill=body, outline=INK, width=4, r=46)
    c.rect(6, 4, length + 80, 36, fill=body_sh, r=26)
    c.rect(4, height * .70, length, height * .70 + 24, fill=C("#2c5d8f"))
    for x in range(1000, length, 1000):
        c.rect(x - 8, 0, x + 8, height - 28, fill=C("#2a2d31"))
    wy0, wy1 = height * .17, height * .56
    rng = np.random.default_rng(length)
    for x in range(150, length, 190):
        if any(d - 140 < x < d + 230 for d in doors) or x % 1000 > 870:
            continue
        fill = C("#f7e7b5") if lit else C("#1e2630")
        c.rect(x, wy0, x + 130, wy1, fill=fill, outline=INK, width=3, r=14)
        if lit:
            if yellow_at is not None and abs(x + 65 - yellow_at) < 95:
                c.ell(x + 65, wy0 + 52, 32, 36, fill=COAT)
                c.ell(x + 65, wy0 + 56, 18, 22, fill=C("#4a3a26"))
                c.rect(x + 22, wy0 + 82, x + 108, wy1, fill=COAT, r=26)
            elif rng.random() < .55:
                cx = x + 65 + rng.integers(-18, 18)
                c.ell(cx, wy0 + 56, 20, 24, fill=C("#3b3127"))
                c.rect(cx - 38, wy0 + 78, cx + 38, wy1, fill=C("#3b3127"), r=22)
        else:
            c.line([(x + 30, wy1 - 10), (x + 80, wy0 + 10)], C("#33404d"), 6)
    for d in doors:
        c.rect(d, 26, d + 220, height - 34, fill=C("#e9dba9") if lit else C("#1a2027"), outline=INK, width=3)
        c.rect(d + 10, height - 50, d + 210, height - 34, fill=C("#3b3f45"))
        if lit:
            for dx in (0, 110):
                c.rect(d + dx, 26, d + dx + 110, height - 34, fill=body, outline=INK, width=3)
                c.rect(d + dx + 22, 50, d + dx + 88, height * .5, fill=C("#f7e7b5"), outline=INK, width=3, r=10)
    c.rect(0, 0, 70, height - 28, fill=body_sh, outline=INK, width=4, r=40)
    c.ell(36, height * .64, 15, 15, fill=C("#fff4c2"), outline=INK, width=2)
    sx = 300
    c.rect(sx, height * .05, sx + 300, height * .05 + 46, fill=C("#0c0c0c"), outline=INK, width=2, r=6)
    c.text(sx + 150, height * .05 + 23, sign, MONO, 30 if len(sign) > 9 else 34, sign_col)
    c.rect(20, height - 40, length + 80, height, fill=C("#141518"))
    for x in range(140, length, 480):
        c.ell(x, height - 16, 32, 20, fill=C("#2b2d31"), outline=C("#55585d"), width=3)
    return c.out()


def door_panel(h):
    c = Cv(110, h)
    c.rect(0, 0, 110, h, fill=C("#b8bdc3"), outline=INK, width=3)
    c.rect(22, 24, 88, h * .45, fill=C("#1e2630"), outline=INK, width=3, r=10)
    return c.out()


INT_WINDOWS = ((40, 400, 500, 900), (580, 400, 1040, 900))


def bg_interior():
    c = Cv(W, H)
    c.rect(0, 0, W, H, fill=C("#b9c2b6"))
    c.rect(0, 0, W, 252, fill=C("#7b827a"))
    c.rect(0, 140, W, 200, fill=C("#f4f8ea"))
    c.rect(0, 200, W, 212, fill=C("#5e655e"))
    c.rect(0, 252, W, 372, fill=C("#d6dad0"))
    ads = [((24, 262, 350, 362), "#1d3557", "NIGHT OWL DINER", "OPEN 24/7"),
           ((376, 262, 704, 362), "#e76f51", "STAND CLEAR", "OF THE DOORS"),
           ((730, 262, 1056, 362), "#2a9d8f", "LOST?", "ASK STATION STAFF")]
    for (x0, y0, x1, y1), col, a, b in ads:
        c.rect(x0, y0, x1, y1, fill=C(col), outline=INK, width=3, r=8)
        c.text((x0 + x1) / 2, y0 + 36, a, BOLD, 30, C("#ffffff"))
        c.text((x0 + x1) / 2, y0 + 74, b, SANS, 22, C("#f0f0f0"))
    for x0, y0, x1, y1 in INT_WINDOWS:
        c.rect(x0 - 16, y0 - 16, x1 + 16, y1 + 16, fill=C("#98a197"), outline=INK, width=3, r=48)
        c.rect(x0, y0, x1, y1, fill=C("#0d1522"), outline=C("#1d1f22"), width=8, r=36)
    c.rect(508, 560, 572, 724, fill=C("#2b2f33"), outline=INK, width=3, r=10)
    c.text(540, 694, "DOORS", BOLD, 13, C("#cfd3d0"))
    c.rect(0, 930, W, 1040, fill=C("#a8b1a5"))
    c.rect(20, 1040, W - 20, 1250, fill=C("#3b5b7a"), outline=C("#22384f"), width=4, r=24)
    for x in (380, 720):
        c.line([(x, 1052), (x, 1240)], C("#2d4660"), 5)
    c.rect(10, 1236, W - 10, 1330, fill=C("#46698c"), outline=C("#22384f"), width=4, r=22)
    c.rect(30, 1330, W - 30, 1460, fill=C("#2a2f33"))
    for x in range(60, W - 60, 28):
        c.line([(x, 1350), (x, 1440)], C("#1d2124"), 6)
    c.rect(0, 1460, W, H, fill=C("#5c6266"))
    rng = np.random.default_rng(4)
    for _ in range(260):
        x, y = rng.integers(0, W), rng.integers(1465, H)
        c.ell(x, y, 3, 2, fill=C("#6f767b") if rng.random() < .5 else C("#4d5357"))
    c.rect(866, 1150, 1000, 1290, fill=TOTE, outline=INK, width=3, r=14)
    c.arc(933, 1150, 50, 60, 180, 360, C("#7a5a1c"), 8)
    for x in (110, 1012):
        c.rect(x - 13, 0, x + 13, H, fill=C("#c9ced2"), outline=C("#6f767c"), width=3)
        c.rect(x - 4, 0, x + 2, H, fill=C("#eef1f3"))
    img = c.out().convert("RGB")

    def g(d):
        d.rectangle([0, 140, W, 200], fill=(70, 80, 60))
    return ImageChops.add(img, glow_layer((W, H), g, 30))


GLASS = (128, 268, 952, 1292)


def bg_window():
    c = Cv(W, H)
    c.rect(0, 0, W, H, fill=C("#a2ab9f"))
    c.rect(0, 0, W, 180, fill=C("#d6dad0"))
    c.rect(120, 26, 960, 154, fill=C("#e76f51"), outline=INK, width=3, r=10)
    c.text(540, 72, "STAND CLEAR", BOLD, 44, C("#ffffff"))
    c.text(540, 122, "OF THE CLOSING DOORS", SANS, 28, C("#fbeee9"))
    c.rect(90, 230, 990, 1330, fill=C("#b4babd"), outline=C("#5b6166"), width=4, r=70)
    c.rect(110, 250, 970, 1310, fill=C("#1d1f22"), r=58)
    c.rect(40, 1380, 1040, 1404, fill=C("#d5d9dc"), outline=C("#6f767c"), width=3, r=12)
    c.rect(0, 1440, W, H, fill=C("#3b5b7a"))
    c.rect(0, 1440, W, 1470, fill=C("#46698c"))
    for x in range(0, W, 180):
        c.line([(x, 1470), (x, H)], C("#2d4660"), 5)
    img = c.out()
    grad = np.zeros((H, W, 4), np.uint8)
    yy = np.linspace(0, 1, H)[:, None]
    grad[..., 0] = (11 + 10 * yy).astype(np.uint8)
    grad[..., 1] = (19 + 14 * yy).astype(np.uint8)
    grad[..., 2] = (32 + 20 * yy).astype(np.uint8)
    grad[..., 3] = 255
    glass = Image.fromarray(grad, "RGBA")
    m = mask_of(lambda c2: c2.rect(*GLASS, fill=C("#ffffff"), r=46))
    img.paste(glass, (0, 0), m)
    return img.convert("RGB"), m


def window_glare(mask):
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    d.polygon([(620, 268), (780, 268), (360, 1292), (200, 1292)], fill=(255, 255, 255, 18))
    d.polygon([(820, 268), (860, 268), (440, 1292), (400, 1292)], fill=(255, 255, 255, 12))
    g = g.filter(ImageFilter.GaussianBlur(10))
    return clip_to(g, mask)


def other_window():
    c = Cv(720, 600)
    c.rect(0, 0, 720, 600, fill=C("#f2e2b0"), outline=C("#6d6a60"), width=22, r=44)
    c.rect(22, 22, 698, 60, fill=C("#fffbea"))
    c.rect(22, 470, 698, 578, fill=C("#5a6f8a"))
    c.rect(590, 22, 606, 578, fill=C("#c9ced2"))
    return c.out()


def phone_closeup():
    c = Cv(W, H)
    c.ell(640, 2080, 420, 330, fill=COAT, outline=INK, width=5)
    c.rect(560, 1500, 760, 1800, fill=SKIN_S, outline=INK, width=4, r=80)
    c.rect(250, 360, 830, 1560, fill=C("#0f1012"), outline=C("#3a3d44"), width=6, r=70)
    c.rect(286, 420, 794, 1500, fill=C("#f7f6f2"), r=44)
    c.text(332, 468, "2:19", SANS, 34, C("#222222"), anchor="lm")
    c.rect(700, 452, 750, 478, outline=C("#222222"), width=3, r=4)
    c.rect(704, 456, 722, 474, fill=C("#222222"))
    c.text(332, 560, "New note", SANS, 30, C("#9a9a9a"), anchor="lm")
    c.line([(320, 600), (760, 600)], C("#e2e2dc"), 3)
    c.text(540, 860, "WRONG", TALL, 165, C("#111111"))
    c.text(540, 1060, "TRAIN", TALL, 165, C("#111111"))
    c.rect(212, 1170, 292, 1320, fill=SKIN_S, outline=INK, width=4, r=38)
    c.rect(206, 1310, 286, 1450, fill=SKIN_S, outline=INK, width=4, r=38)
    c.rect(788, 1230, 872, 1450, fill=SKIN_S, outline=INK, width=4, r=40)
    img = c.out()
    tw = font(165, TALL).getlength("TRAIN")
    return img, (540 + tw / 2 + 10, 995, 540 + tw / 2 + 20, 1125)


def led_matrix(text, pitch=8, rad=3.2, on=(255, 59, 48), off=(58, 13, 11)):
    f = font(14, MONO)
    tmp = Image.new("L", (int(f.getlength(text)) + 8, 24), 0)
    d = ImageDraw.Draw(tmp)
    d.fontmode = "1"
    d.text((4, 2), text, font=f, fill=255)
    arr = np.array(tmp) > 127
    rows = np.where(arr.any(1))[0]
    arr = arr[max(rows[0] - 1, 0):rows[-1] + 2]
    h, w = arr.shape
    c = Cv(w * pitch, h * pitch)
    for y in range(h):
        for x in range(w):
            c.ell(x * pitch + pitch / 2, y * pitch + pitch / 2, rad, rad,
                  fill=on + (255,) if arr[y, x] else off + (255,))
    return c.out()


TILT_H = 3000


def tilt_composite():
    c = Cv(W, TILT_H)
    c.rect(0, 0, W, TILT_H, fill=C("#b9c2b6"))
    c.rect(0, 0, W, 250, fill=C("#7b827a"))
    c.rect(0, 120, W, 170, fill=C("#f4f8ea"))
    c.rect(60, 320, 1020, 570, fill=C("#101010"), outline=C("#3a3a3a"), width=6, r=18)
    c.rect(180, 610, 900, 1820, fill=C("#98a197"), outline=INK, width=4, r=16)
    for x0 in (200, 542):
        c.rect(x0, 630, x0 + 338, 1820, fill=C("#c4cac4"), outline=INK, width=4)
        c.rect(x0 + 50, 710, x0 + 288, 1150, fill=C("#0d1522"), outline=INK, width=4, r=20)
    c.rect(0, 1820, W, TILT_H, fill=C("#5c6266"))
    c.ell(540, 3020, 470, 420, fill=JACKET, outline=INK, width=6)
    c.poly([(440, 2620), (640, 2620), (600, TILT_H), (480, TILT_H)], fill=SCRUB)
    c.rect(480, 2560, 600, 2680, fill=SKIN_M, outline=INK, width=4)
    draw_maya_head(c, 540, 2330, 250, "shock", (0, -1))
    img = c.out()
    led = led_matrix("NOT IN SERVICE")
    comp(img, led, 540 - led.width / 2, 445 - led.height / 2)
    return img.convert("RGB"), (540 - led.width / 2, 445 - led.height / 2, led.width, led.height)


# ---- aerial
def bezier(p0, p1, p2, p3, n=500):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def make_path(x, c1, c2, end, y0=2100, y1=1150):
    straight = np.stack([np.full(200, x), np.linspace(y0, y1, 200)], 1)
    curve = bezier((x, y1), c1, c2, end)
    pts = np.vstack([straight, curve[1:]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return pts, np.concatenate([[0], np.cumsum(seg)])


PATH_L = make_path(470, (470, 820), (330, 470), (-200, 250))
PATH_R = make_path(610, (610, 860), (760, 700), (1300, 640))


def path_at(path, s):
    pts, S = path
    s = clamp(s, 0, S[-1] - 1e-3)
    i = int(np.searchsorted(S, s)) - 1
    i = max(0, min(i, len(S) - 2))
    f = (s - S[i]) / max(S[i + 1] - S[i], 1e-6)
    p = pts[i] + (pts[i + 1] - pts[i]) * f
    d = pts[i + 1] - pts[i]
    return p[0], p[1], math.atan2(d[1], d[0])


def oriented_rect(x, y, ang, length, width):
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    hl, hw = length / 2, width / 2
    return [(x + ux * hl + nx * hw, y + uy * hl + ny * hw), (x + ux * hl - nx * hw, y + uy * hl - ny * hw),
            (x - ux * hl - nx * hw, y - uy * hl - ny * hw), (x - ux * hl + nx * hw, y - uy * hl + ny * hw)]


def bg_aerial():
    rng = np.random.default_rng(7)
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (17, 21, 25)
    base += rng.normal(0, 9, (H, W, 1))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    c = Cv(W, H)
    for path in (PATH_L, PATH_R):
        S = path[1][-1]
        for s in np.arange(0, S, 26):
            x, y, a = path_at(path, s)
            c.poly(oriented_rect(x, y, a, 12, 118), fill=C("#2e2620"))
    for path in (PATH_L, PATH_R):
        S = path[1][-1]
        for off in (-38, 38):
            pts = []
            for s in np.arange(0, S, 10):
                x, y, a = path_at(path, s)
                pts.append((x - math.sin(a) * off, y + math.cos(a) * off))
            c.line(pts, C("#7c828a"), 5)
    c.rect(220, 265, 440, 335, fill=C("#1f7a45"), outline=C("#d9f2e2"), width=3, r=10)
    c.text(330, 300, "DOWNTOWN", BOLD, 30, C("#ffffff"))
    c.rect(720, 525, 980, 595, fill=C("#b5651d"), outline=C("#ffe0b0"), width=3, r=10)
    c.text(850, 560, "RAIL YARD", BOLD, 30, C("#ffffff"))
    c.rect(745, 600, 955, 632, fill=C("#3a1f0b"), r=6)
    c.text(850, 616, "NO PASSENGERS", BOLD, 17, C("#ffcf9a"))
    img.alpha_composite(c.out())
    return img.convert("RGB")


def title_card():
    c = Cv(W, H)
    c.rect(0, 0, W, H, fill=C("#050608"))
    c.text(540, 840, "THE LAST TRAIN", BOLD, 100, C("#f2f2f2"))
    c.rect(300, 912, 780, 918, fill=C("#c0392b"))
    c.text(540, 1010, "AT 2:17 AM", BOLD, 128, C("#ffb020"))
    return c.out().convert("RGB")


def vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 * 0.8 + ((yy - H / 2) / (H / 2)) ** 2)
    a = np.clip((r - 0.62) / 0.7, 0, 1) ** 1.4 * 200
    arr = np.zeros((H, W, 4), np.uint8)
    arr[..., 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def caption_img(text, hl):
    f = font(76, BOLD)
    lines, cur = [], ""
    for w in text.split():
        test = (cur + " " + w).strip()
        if f.getlength(test) <= 900 or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    lh = 98
    im = Image.new("RGBA", (W, lh * len(lines) + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text((W / 2, 20 + lh * i + lh / 2), ln, font=f, anchor="mm",
               fill=(255, 210, 63, 255) if hl else (255, 255, 255, 255),
               stroke_width=9, stroke_fill=(0, 0, 0, 255))
    return im


A = {}


def build_assets():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    A["platform"] = bg_platform()
    A["near_train"] = train_side(2400, 440, False, "NOT IN SERVICE", C("#ff4136"), [980])
    A["far_train"] = train_side(2200, 300, True, "DOWNTOWN", C("#3ddc84"), [700, 1700], yellow_at=960)
    A["door_panel"] = door_panel(440 - 26 - 34)
    A["interior"] = bg_interior()
    A["int_mask"] = mask_of(lambda c: [c.rect(*w, fill=C("#ffffff"), r=36) for w in INT_WINDOWS])
    A["window"], A["glass_mask"] = bg_window()
    A["glare"] = window_glare(A["glass_mask"])
    A["otherwin"] = other_window()
    A["phone"], A["cursor"] = phone_closeup()
    A["tilt"], A["led_box"] = tilt_composite()
    A["aerial"] = bg_aerial()
    A["title"] = title_card()
    A["vignette"] = vignette()
    pg = Image.new("RGBA", (420, 520), (0, 0, 0, 0))
    ImageDraw.Draw(pg).ellipse([90, 90, 330, 430], fill=(200, 225, 255, 120))
    A["phone_glow"] = pg.filter(ImageFilter.GaussianBlur(40))

    exports = {
        "bg_platform.png": A["platform"], "bg_interior.png": A["interior"], "bg_window.png": A["window"],
        "bg_door_sign_tilt.png": A["tilt"], "bg_aerial_tracks.png": A["aerial"], "end_card.png": A["title"],
        "prop_train_near_out_of_service.png": A["near_train"], "prop_train_far_downtown.png": A["far_train"],
        "prop_door_panel.png": A["door_panel"], "prop_other_train_window.png": A["otherwin"],
        "prop_phone_wrong_train.png": A["phone"], "maya_body_seated.png": maya_body(),
        "maya_standing_step0.png": maya_standing(0), "maya_standing_step1.png": maya_standing(1),
        "stranger_calm.png": stranger("calm"), "stranger_urgent_phone.png": stranger("urgent", True),
    }
    for e in ("tired", "uneasy", "scared", "shock", "dread"):
        exports[f"maya_head_{e}.png"] = maya_head(e)
    exports["maya_head_turned_scared.png"] = maya_head("scared", (-1, 0), -1.0)
    for name, img in exports.items():
        img.save(ART_DIR / name)


# --------------------------------------------------------------------------- timeline
def plan_timeline(durs):
    tl, t = [], 0.0
    for sid, dur, lines in SCENES:
        lead = 0.12 if sid == "s1" else 0.3
        tail = 0.3
        ds = [durs[f"{sid}_{i}"] for i in range(len(lines))]
        pres = sum(ln["pre"] for ln in lines)
        n, min_gap = len(ds), 0.2
        avail = dur - lead - tail - pres
        need = sum(ds) + min_gap * (n - 1)
        speed = max(SLOW, need / avail) if sid != "s8" else SLOW
        ds2 = [d / speed for d in ds]
        gap = max(min_gap, min(0.7, (avail - sum(ds2)) / (n - 1))) if n > 1 else 0
        cur, L_ = t + lead, []
        for i, (d, ln) in enumerate(zip(ds2, lines)):
            cur += ln["pre"]
            L_.append(dict(id=f"{sid}_{i}", cap=ln["cap"], hl=ln["hl"], start=cur, end=cur + d))
            cur += d + gap
        if sid == "s8":
            dur = min(max(dur, L_[-1]["end"] - t + 0.45 + 1.9), MAX_TOTAL - t)
        tl.append(dict(id=sid, start=t, dur=dur, speed=speed, lines=L_))
        t += dur
    return tl


def make_events(TL):
    s = {sc["id"]: sc for sc in TL}

    def st(i):
        return s[i]["start"]

    def ls(sid, i):
        return s[sid]["lines"][i]["start"]

    def le(sid, i):
        return s[sid]["lines"][i]["end"]

    walk0 = min(ls("s2", 2) - 0.2, st("s2") + s["s2"]["dur"] - 3.0)
    E = dict(
        s1_flicker=st("s1") + 2.7,
        s2_screech=st("s2") + 1.2, s2_chime=st("s2") + 2.85, s2_open=st("s2") + 3.15,
        s2_walk0=walk0, s2_walk1=walk0 + 1.9, s2_chime2=walk0 + 2.05, s2_close=walk0 + 2.45,
        s3_cut=ls("s3", 1) - 0.15,
        s4_reveal=ls("s4", 0) + 0.45, s4_turn=ls("s4", 2) - 0.12, s4_pan=ls("s4", 3) - 0.25,
        s5_flicks=[st("s5") + x for x in (0.5, 0.78, 1.4, 1.6)],
        s5_tap=ls("s5", 1) + 1.5,
        s6_tilt=ls("s6", 2) - 0.1, s6_sign=ls("s6", 3) - 0.05,
        s7_switch=st("s7") + 0.25, s7_aerial=ls("s7", 2) - 0.15,
        s8_lock=ls("s8", 1), s8_dark=le("s8", 2) + 0.12,
        s8_cut=ls("s8", 3) - 0.2,
        title=le("s8", 3) + 0.45,
        end=st("s8") + s["s8"]["dur"],
    )
    return E


# --------------------------------------------------------------------------- audio
RNG = np.random.default_rng(217)


def tt(d):
    return np.arange(int(d * SR)) / SR


def filt(x, lo=None, hi=None, order=2):
    pad = int(0.3 * SR)
    n = len(x) + pad
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g /= np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    if hi:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, n)[:len(x)]


def wn(d):
    return RNG.standard_normal(int(d * SR))


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x / m * peak if m > 0 else x


def fade(x, a=0.01, b=0.05):
    x = x.copy()
    na, nb = min(int(a * SR), len(x)), min(int(b * SR), len(x))
    if na:
        x[:na] *= np.linspace(0, 1, na)
    if nb:
        x[-nb:] *= np.linspace(1, 0, nb)
    return x


def sfx_boom(d=3.5):
    t = tt(d)
    f = 30 + 50 * np.exp(-t * 1.6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.1)
    x += filt(wn(d), hi=180) * np.exp(-t * 2.5) * 0.9
    x += filt(wn(d), lo=800, hi=5000) * np.exp(-t * 35) * 0.25
    return norm(fade(x, 0.002, 0.3))


def sfx_riser(d):
    t = tt(d)
    x = filt(wn(d), lo=400, hi=7000) * (t / d) ** 2.2
    x += 0.4 * np.sin(2 * np.pi * np.cumsum(180 + 700 * (t / d) ** 2) / SR) * (t / d) ** 2
    return norm(fade(x, 0.2, 0.03))


def sfx_heartbeat():
    def thump(g):
        t = tt(0.25)
        return g * np.sin(2 * np.pi * 52 * t) * np.exp(-t * 22) + g * 0.3 * filt(wn(0.25), hi=150) * np.exp(-t * 30)
    x = np.zeros(int(0.7 * SR))
    a, b = thump(1.0), thump(0.65)
    x[:len(a)] += a
    i = int(0.24 * SR)
    x[i:i + len(b)] += b
    return norm(x)


def sfx_train_bed(d, period=0.92):
    t = tt(d)
    x = filt(wn(d), hi=110) * 1.6 + filt(wn(d), lo=180, hi=700) * 0.22
    x *= 1 + 0.18 * np.sin(2 * np.pi * 0.6 * t)
    x += 0.25 * np.sin(2 * np.pi * 47 * t)
    burst = filt(wn(0.09), lo=250, hi=2600) * np.exp(-tt(0.09) * 55)
    k = 0.0
    while k < d - 0.3:
        for off, g in ((0, 1.0), (0.13, 0.8)):
            i = int((k + off) * SR)
            x[i:i + len(burst)] += burst * g * 2.2
        k += period
    return norm(x)


def sfx_arrival(d=3.4):
    t = tt(d)
    env = np.clip(t / 1.6, 0, 1) ** 1.5 * np.exp(-np.clip(t - 2.4, 0, None) * 2.2)
    x = (filt(wn(d), hi=160) * 1.5 + filt(wn(d), lo=200, hi=1500) * 0.3) * env
    return norm(x)


def sfx_screech(d=1.7):
    t = tt(d)
    env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5
    x = sum(np.sin(2 * np.pi * np.cumsum(f + 40 * np.sin(2 * np.pi * 7 * t)) / SR) * g
            for f, g in ((2900, 1), (3420, .6), (4100, .35)))
    x = x * env + filt(wn(d), lo=2000, hi=6000) * env * 0.4
    return norm(x)


def sfx_chime():
    out = np.zeros(int(1.8 * SR))
    for start, f in ((0, 659.25), (0.42, 523.25)):
        t = tt(1.3)
        x = sum(g * np.sin(2 * np.pi * f * m * t) for m, g in ((1, 1), (2, .3), (3, .1))) * np.exp(-t * 3.2)
        i = int(start * SR)
        out[i:i + len(x)] += x
    return norm(fade(out, 0.003, 0.1))


def sfx_hiss(d=0.7):
    t = tt(d)
    return norm(filt(wn(d), lo=2500, hi=9000) * np.exp(-t * 5) * np.clip(t / 0.03, 0, 1))


def sfx_thud():
    t = tt(0.5)
    x = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 14) + filt(wn(0.5), hi=400) * np.exp(-t * 20) * 0.6
    return norm(x)


def sfx_buzz(d):
    t = tt(d)
    x = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 9))
    return norm(filt(x, hi=3000))


def sfx_zap():
    t = tt(0.16)
    x = (filt(wn(0.16), lo=1500) * 0.6 + sfx_buzz(0.16) * 0.8) * np.exp(-t * 18)
    return norm(x)


def sfx_tap():
    out = np.zeros(int(0.5 * SR))
    t = tt(0.25)
    one = filt(wn(0.25), lo=1500, hi=6000) * np.exp(-t * 90) + np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 45) * 0.3
    for s in (0.0, 0.2):
        i = int(s * SR)
        out[i:i + len(one)] += one
    return norm(out)


def sfx_blip():
    t = tt(0.09)
    x = np.concatenate([np.sin(2 * np.pi * 1320 * t), np.sin(2 * np.pi * 1760 * t)])
    return norm(fade(x, 0.003, 0.03))


def sfx_clank():
    t = tt(1.4)
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * dcy) * g for f, dcy, g in
            ((420, 4, 1), (1063, 6, .7), (1720, 8, .5), (2611, 11, .35), (3307, 14, .25)))
    x += filt(wn(1.4), lo=500, hi=5000) * np.exp(-t * 30) * 1.2
    return norm(x)


def sfx_whoosh(d=2.2):
    t = tt(d)
    env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 2
    lo = filt(wn(d), lo=150, hi=900)
    hi = filt(wn(d), lo=900, hi=4000)
    mix = np.clip(t / d, 0, 1)
    return norm((lo * (1 - mix) + hi * mix) * env)


def sfx_lock():
    x = sfx_thud() * 0.8
    click = filt(wn(0.03), lo=3000) * np.exp(-tt(0.03) * 200)
    x[:len(click)] += click * 0.9
    return norm(x)


def sfx_powerdown(d=1.8):
    t = tt(d)
    f = 30 + 330 * np.exp(-t * 2.2)
    x = (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * np.sin(4 * np.pi * np.cumsum(f) / SR)) * np.exp(-t * 1.4)
    return norm(fade(x, 0.01, 0.2))


CHORDS = {
    "Am": (110.0, 164.81, 220.0, 261.63), "F": (87.31, 130.81, 174.61, 220.0),
    "Dm": (73.42, 146.83, 174.61, 220.0), "Bb": (116.54, 174.61, 233.08, 293.66),
    "E": (82.41, 123.47, 164.81, 207.65),
}


def music_block(chord, d):
    t = tt(d + 1.2)
    L_ = np.zeros(len(t))
    R_ = np.zeros(len(t))
    for f in CHORDS[chord]:
        for dt, lr in ((-0.004, 0), (0.0, 1), (0.004, 2)):
            ph = RNG.random() * 6.28
            v = np.sin(2 * np.pi * f * (1 + dt) * t + ph) + 0.25 * np.sin(4 * np.pi * f * (1 + dt) * t + ph)
            if lr == 0:
                L_ += v
            elif lr == 2:
                R_ += v
            else:
                L_ += v * .7
                R_ += v * .7
    trem = 1 + 0.12 * np.sin(2 * np.pi * 0.25 * t)
    env = np.minimum(np.clip(t / 0.6, 0, 1), np.clip((t[-1] - t) / 0.6, 0, 1))
    return np.stack([filt(L_, hi=900), filt(R_, hi=900)]) * trem * env


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        data = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
    if ch > 1:
        data = data.reshape(-1, ch).mean(1)
    assert sr == SR, f"{p} is {sr} Hz"
    return data


def write_wav(p, x):
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    pcm = (np.clip(x.T, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def trim(x, thr=0.008):
    idx = np.where(np.abs(x) > thr)[0]
    if not len(idx):
        return x
    a, b = max(idx[0] - int(0.03 * SR), 0), min(idx[-1] + int(0.08 * SR), len(x))
    return x[a:b]


def voice_fx(x, speed):
    idx = np.arange(0, len(x) - 1, speed)
    x = np.interp(idx, np.arange(len(x)), x)
    x = filt(x, lo=85)
    x = x + 0.25 * filt(x, lo=2500, hi=6000)
    ir_t = tt(0.55)
    ir = RNG.standard_normal(len(ir_t)) * np.exp(-ir_t * 9)
    ir = filt(ir, lo=300, hi=4000)
    wet = np.fft.irfft(np.fft.rfft(x, len(x) + len(ir)) * np.fft.rfft(ir, len(x) + len(ir)))[:len(x) + len(ir) // 2]
    out = np.zeros(len(wet))
    out[:len(x)] += x
    out += norm(wet, 0.13 * np.max(np.abs(x)))
    return norm(out, 0.9)


def make_narration():
    NARR_DIR.mkdir(parents=True, exist_ok=True)
    job = []
    for sid, _, lines in SCENES:
        for i, ln in enumerate(lines):
            job.append(dict(id=f"{sid}_{i}", text=ln["tts"], path=str(NARR_DIR / f"{sid}_{i}_raw.wav")))
    jf = NARR_DIR / "_job.json"
    jf.write_text(json.dumps(job), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ROOT / "build" / "tts.ps1"), "-Job", str(jf)], check=True)
    jf.unlink()
    raw = {j["id"]: trim(read_wav(j["path"])) for j in job}
    return raw


def build_audio(TL, E, raw):
    total = E["end"]
    n = int((total + 0.5) * SR)
    voice = np.zeros(n)
    fx = np.zeros((2, n))
    mus = np.zeros((2, n))
    time = np.arange(n) / SR

    def add(buf, x, t, g=1.0, pan=0.0):
        i = int(t * SR)
        if i >= n or t < 0:
            return
        x = x[:n - i]
        if buf.ndim == 1:
            buf[i:i + len(x)] += g * x
        else:
            buf[0, i:i + len(x)] += g * x * math.cos((pan + 1) * math.pi / 4) * 1.414
            buf[1, i:i + len(x)] += g * x * math.sin((pan + 1) * math.pi / 4) * 1.414

    for sc in TL:
        for ln in sc["lines"]:
            clip = voice_fx(raw[ln["id"]], sc["speed"])
            NARR_DIR.mkdir(parents=True, exist_ok=True)
            write_wav(NARR_DIR / f"{ln['id']}.wav", clip)
            add(voice, clip, ln["start"])

    st = {sc["id"]: sc["start"] for sc in TL}

    def curve(points):
        xs, ys = zip(*points)
        return np.interp(time, xs, ys)

    bed = sfx_train_bed(total + 0.5)[:n]
    g_int = curve([(0, .9), (st["s2"] - .05, .9), (st["s2"] + .05, 0), (st["s3"] - .05, 0), (st["s3"] + .15, .9),
                   (E["s8_dark"], .9), (E["s8_dark"] + 1.2, .35), (E["title"], .15), (total, 0)])
    plat = filt(wn(total + 0.5), lo=60, hi=500)[:n]
    plat = norm(plat) * 0.5
    g_plat = curve([(0, 0), (st["s2"] - .01, 0), (st["s2"] + .1, 1), (st["s3"] - .1, 1), (st["s3"], 0), (total, 0)])
    add(fx, bed * g_int * 0.26, 0)
    add(fx, plat * g_plat * 0.12, 0)
    hum = sfx_buzz(total + 0.5)[:n] * curve([(0, 1), (st["s2"], 1), (st["s2"] + .01, 0), (st["s3"], 0),
                                              (st["s3"] + .01, 1), (E["s8_dark"], 1), (E["s8_dark"] + .4, 0), (total, 0)])
    add(fx, hum * 0.018, 0)

    add(fx, sfx_boom(), 0.0, 0.9)
    add(fx, sfx_zap(), E["s1_flicker"], 0.35, 0.3)
    add(fx, sfx_arrival(), st["s2"] - 0.2, 0.5, 0.4)
    add(fx, sfx_screech(), E["s2_screech"], 0.12, 0.5)
    add(fx, sfx_chime(), E["s2_chime"], 0.22)
    add(fx, sfx_hiss(), E["s2_open"], 0.18)
    add(fx, sfx_chime(), E["s2_chime2"], 0.2)
    add(fx, sfx_thud(), E["s2_close"] + 0.45, 0.5)
    add(fx, sfx_boom(2.5), E["s4_reveal"], 0.45)
    hb = sfx_heartbeat()
    t, end_hb = E["s4_reveal"] + 0.3, st["s7"] + 0.4
    while t < end_hb:
        p = clamp((t - E["s4_reveal"]) / (end_hb - E["s4_reveal"]))
        add(fx, hb, t, 0.55 + 0.3 * p)
        t += 60 / lerp(72, 112, p)
    for ft in E["s5_flicks"]:
        add(fx, sfx_zap(), ft, 0.45, RNG.uniform(-.4, .4))
    add(fx, sfx_riser(E["s6_sign"] - st["s5"]), st["s5"], 0.28)
    add(fx, sfx_tap(), E["s5_tap"], 0.5, -0.2)
    add(fx, sfx_blip(), st["s6"] + 0.15, 0.12)
    add(fx, sfx_buzz(E["end"] - E["s6_tilt"])[: int((st["s7"] - E["s6_tilt"]) * SR)], E["s6_tilt"], 0.03)
    add(fx, sfx_boom(), E["s6_sign"], 1.0)
    add(fx, sfx_clank(), E["s7_switch"], 0.45, 0.2)
    add(fx, sfx_whoosh(), E["s7_aerial"], 0.35)
    add(fx, sfx_lock(), E["s8_lock"], 0.55, -0.3)
    add(fx, sfx_lock(), E["s8_lock"] + 0.28, 0.5, 0.3)
    add(fx, sfx_powerdown(), E["s8_dark"], 0.45)
    add(fx, sfx_boom(4), E["title"], 0.9)

    plan = [("s1", "Am", .55), ("s2", "F", .3), ("s3", "Am", .35), ("s4", "Dm", .55), ("s5", "Bb", .7),
            ("s6", "E", .85), ("s7", "F", .5), ("s8", "Dm", .6)]
    for sid, ch, g in plan:
        sc = next(s for s in TL if s["id"] == sid)
        blk = music_block(ch, sc["dur"])
        add(mus, blk[0] * g, sc["start"] - 0.6, 1.0, -1)
        add(mus, blk[1] * g, sc["start"] - 0.6, 1.0, 1)
    mus *= curve([(0, 1), (E["s8_cut"] - .4, 1), (E["s8_cut"], 0), (E["title"], 0), (E["title"] + .1, .8), (total, .8)])
    tail = music_block("Am", total - E["title"] + 0.5)
    add(mus, tail[0] * 0.7, E["title"], 1.0, -1)
    add(mus, tail[1] * 0.7, E["title"], 1.0, 1)
    eerie = np.sin(2 * np.pi * np.cumsum(1760 + 6 * np.sin(2 * np.pi * 5 * time)) / SR)
    add(mus, eerie * curve([(0, 0), (st["s4"], 0), (st["s4"] + 1, .05), (st["s7"], .05), (st["s7"] + .5, 0), (total, 0)]), 0)

    k = int(0.15 * SR)
    env = np.convolve(np.abs(voice), np.ones(k) / k, "same")
    duck = 1 - 0.55 * np.clip(env / 0.06, 0, 1)
    mus = norm(mus, 1.0) * 0.16 * duck
    fx = fx * (1 - 0.3 * np.clip(env / 0.06, 0, 1))
    v2 = np.vstack([voice, voice])
    # Stems keep their mixed levels, so V+SFX+music on a Shotcut timeline == full_mix.
    g = 0.9 / max(np.max(np.abs(v2 + fx + mus)), 1e-6)
    stems = {"narration_track.wav": v2 * g, "sfx_track.wav": fx * g, "music_track.wav": mus * g}
    for name, x in stems.items():
        write_wav(AUDIO_DIR / name, x)
    write_wav(AUDIO_DIR / "full_mix.wav", sum(stems.values()))
    return {k: AUDIO_DIR / k for k in stems}, AUDIO_DIR / "full_mix.wav"


# --------------------------------------------------------------------------- rendering
def window_shot(t, maya_expr="uneasy", man=None, man_alpha=1.0, man_scale=0.72, man_pos=(450, 600),
                phone=False, other_alpha=0.22, drift=(0, 0), streak=1.0, refl_scale=1.0, maya_gaze=(0, 0)):
    frame = A["window"].convert("RGBA")
    glass = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(glass)
    for i in range(8):
        y = 300 + i * 128 + (i * 37) % 60
        x = 1100 - ((t * 1900 * streak + i * 530) % 2600)
        d.rounded_rectangle([x, y, x + 160 + (i % 3) * 90, y + 6], radius=3, fill=(255, 220, 160, 70))
    if man:
        sx, sy = math.sin(t * 2.3) * 8, math.sin(t * 3.1 + 1) * 5
        mx, my = man_pos[0] + drift[0] + sx, man_pos[1] + drift[1] + sy
        comp(glass, A["otherwin"], 420 + drift[0] + sx - 360, 700 + drift[1] + sy - 300, other_alpha)
        img = scaled(stranger(man, phone), man_scale)
        comp(glass, img, mx - 300 * man_scale, my - 282 * man_scale, man_alpha)
        if phone:
            gl = scaled(A["phone_glow"], man_scale)
            comp(glass, gl, mx + (305 - 300) * man_scale - gl.width / 2,
                 my + (550 - 282) * man_scale - gl.height / 2, man_alpha)
    if maya_expr:
        r = scaled(maya_reflection(maya_expr, maya_gaze), refl_scale)
        comp(glass, r, 610 - 300 * refl_scale, 880 - 230 * refl_scale)
    clip_to(glass, A["glass_mask"])
    frame.alpha_composite(glass)
    frame.alpha_composite(A["glare"])
    return frame.convert("RGB")


def interior_shot(t, expr="tired", gaze=(0, 0), turn=0.0, tilt=0.0, lamp="green", lights=None, yard=False):
    frame = A["interior"].convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if yard:
        for i in range(3):
            x = 1200 - ((t * 260 + i * 700) % 2100)
            d.ellipse([x, 520 + i * 90, x + 220, 640 + i * 90], fill=(255, 150, 40, 90))
    else:
        for i in range(6):
            y = 430 + i * 80
            x = 1100 - ((t * 2100 + i * 610) % 2600)
            d.rounded_rectangle([x, y, x + 180, y + 7], radius=3, fill=(255, 230, 170, 110))
    if yard:
        layer = layer.filter(ImageFilter.GaussianBlur(25))
    clip_to(layer, A["int_mask"])
    frame.alpha_composite(layer)
    comp(frame, maya_seated(expr, gaze, turn, tilt), 380, 550)
    col = {"green": (60, 220, 110), "red": (255, 50, 40)}[lamp]
    ImageDraw.Draw(frame).ellipse([524, 584, 556, 616], fill=col + (255,))
    g = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse([20, 20, 100, 100], fill=col + (120,))
    comp(frame, g.filter(ImageFilter.GaussianBlur(14)), 480, 540)
    return frame.convert("RGB")


def r_s1(rel, sc):
    img = window_shot(rel, "uneasy", "calm", 1.0)
    if 0 <= sc["start"] + rel - E["s1_flicker"] < 0.1:
        img = darken(img, 0.15)
    return camera(img, lerp(1.0, 1.14, ease(rel / sc["dur"])), 520, 760)


def r_s2(rel, sc):
    frame = A["platform"].convert("RGBA")
    xf = lerp(1150, -400, ease_out((rel - 0.25) / 2.9))
    xn = lerp(1150, -530, ease_out(rel / 2.8))
    comp(frame, A["far_train"], xf, 690)
    comp(frame, A["near_train"], xn, 1000)
    t_abs = sc["start"] + rel
    o = 100 * ease((t_abs - E["s2_open"]) / 0.45) * (1 - ease((t_abs - E["s2_close"]) / 0.45))
    dp = A["door_panel"]
    dx0, dy = xn + 980, 1000 + 26
    if o < 109:
        frame.alpha_composite(dp, (int(max(dx0, 0)), dy), (int(o), 0, 110, dp.height)) if dx0 >= 0 else None
        if dx0 + 110 + o >= 0:
            comp(frame, dp.crop((0, 0, max(1, int(110 - o)), dp.height)), dx0 + 110 + o, dy)
    p = clamp((t_abs - E["s2_walk0"]) / (E["s2_walk1"] - E["s2_walk0"]))
    if p < 1:
        x, y = lerp(800, 560, ease(p)), lerp(1770, 1460, ease(p))
        s = lerp(1.0, 0.78, ease(p))
        moving = 0 < p < 1
        step = int(rel * 4) % 2 if moving else 0
        bob = abs(math.sin(rel * 2 * math.pi * 2)) * 10 if moving else 0
        m = scaled(maya_standing(step), s)
        alpha = 1 - ease((p - 0.82) / 0.18)
        comp(frame, m, x - m.width / 2, y - m.height + 10 - bob, alpha)
    img = frame.convert("RGB")
    img = camera(img, lerp(1.0, 1.05, rel / sc["dur"]), 540, 1000)
    if rel < 2.4:
        ov = img.convert("RGBA")
        lab = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
        ImageDraw.Draw(lab).text((W / 2, 60), "10 MINUTES EARLIER", font=font(46, BOLD), anchor="mm",
                                 fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0, 255))
        comp(ov, lab, 0, 150, min(1, rel / 0.3) * (1 - ease((rel - 1.9) / 0.5)))
        img = ov.convert("RGB")
    return img


def r_s3(rel, sc):
    t_abs = sc["start"] + rel
    if t_abs < E["s3_cut"]:
        tilt = -4 + 3 * math.sin(rel * 1.7)
        img = interior_shot(rel, "tired", (0, 0.4), 0, tilt)
        return camera(img, lerp(1.0, 1.08, rel / 4), 600 + 8 * math.sin(rel * 1.3), 980)
    k = t_abs - E["s3_cut"]
    img = window_shot(rel, "tired")
    return camera(img, lerp(1.0, 1.06, k / 3), 620, 820)


def r_s4(rel, sc):
    t_abs = sc["start"] + rel
    if t_abs < E["s4_turn"]:
        a = ease((t_abs - E["s4_reveal"]) / 0.6)
        expr = "scared" if t_abs > E["s4_reveal"] + 0.7 else "tired"
        img = window_shot(rel, expr, "calm", a)
        k = rel
        return camera(img, lerp(1.06, 1.2, ease(k / 5)), lerp(620, 480, ease(k / 5)), lerp(820, 700, ease(k / 5)))
    img = interior_shot(rel, "scared", (-1, 0), -1.0, 3)
    p = ease((t_abs - E["s4_pan"]) / 0.7)
    return camera(img, lerp(1.25, 1.3, p), lerp(680, 300, p), lerp(900, 1120, p))


def s5_state(t_abs):
    n = sum(1 for f in E["s5_flicks"] if t_abs >= f + 0.09)
    black = any(f <= t_abs < f + 0.09 for f in E["s5_flicks"])
    return n, black


def r_s5(rel, sc):
    t_abs = sc["start"] + rel
    n, black = s5_state(t_abs)
    scale = [0.72, 0.88, 1.02, 1.16, 1.26][n] + (0.05 * ease((rel - 2) / 4) if n >= 4 else 0)
    pos = [(450, 600), (455, 600), (460, 600), (465, 600), (470, 600)][n]
    expr = "urgent" if n >= 2 else "calm"
    img = window_shot(rel, "scared", expr, 1.0, scale, pos, phone=n >= 4, refl_scale=0.92)
    lvl = 0.08 if black else 0.85 + 0.15 * math.sin(rel * 37) * math.sin(rel * 13)
    img = darken(img, lvl)
    return camera(img, 1.08, 540, 760)


def r_s6(rel, sc):
    t_abs = sc["start"] + rel
    if t_abs < E["s6_tilt"]:
        frame = window_shot(rel, None, streak=0.8).convert("RGBA")
        frame = camera(frame.convert("RGB"), 1.5, 540, 780).convert("RGBA")
        ph = A["phone"].rotate(-3, resample=BC, center=(540, 960))
        frame.alpha_composite(ph)
        if int(rel * 2.4) % 2 == 0:
            ImageDraw.Draw(frame).rectangle(A["cursor"], fill=(20, 120, 255, 255))
        img = frame.convert("RGB")
        return camera(img, lerp(1.0, 1.12, ease(rel / 4)), 540, 900)
    p = ease((t_abs - E["s6_tilt"]) / 1.6)
    y0 = lerp(TILT_H - H, 0, p)
    img = A["tilt"].crop((0, int(y0), W, int(y0) + H))
    if t_abs >= E["s6_sign"]:
        k = t_abs - E["s6_sign"]
        x, y, w, h = A["led_box"]
        g = Image.new("RGB", (W, H), (0, 0, 0))
        ImageDraw.Draw(g).rectangle([x, y - y0, x + w, y + h - y0], fill=(255, 40, 30))
        g = g.filter(ImageFilter.GaussianBlur(40))
        g = ImageEnhance.Brightness(g).enhance(0.6 + 0.6 * math.exp(-k * 3) + 0.15 * (int(k * 4) % 2))
        img = ImageChops.add(img, g)
        shake = math.exp(-k * 5) * 18
        img = camera(img, 1.04 + 0.04 * ease(k / 2), 540 + shake * math.sin(k * 60), 600 + shake * math.cos(k * 47))
    return img


def r_s7(rel, sc):
    t_abs = sc["start"] + rel
    if t_abs < E["s7_aerial"]:
        p = ease_out((rel - 0.25) / (E["s7_aerial"] - sc["start"] - 0.25))
        drift = (-950 * p ** 1.3, -140 * p)
        img = window_shot(rel, "dread", "urgent", 1.0, 1.1, (470, 600), phone=False,
                          other_alpha=lerp(0.3, 0.75, ease(rel / 0.8)), drift=drift, streak=0.7)
        return camera(img, 1.05, 540, 760)
    k = t_abs - E["s7_aerial"]
    frame = A["aerial"].copy()
    c = Cv(W, H)
    glow = Image.new("RGB", (W // 4, H // 4), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for path, lit in ((PATH_L, True), (PATH_R, False)):
        head = 620 + 330 * k + (0 if lit else -40)
        for i in range(5):
            s = head - i * 222 - 105
            x, y, a = path_at(path, s)
            if lit:
                c.poly(oriented_rect(x, y, a, 210, 80), fill=C("#c7ccd2"), outline=INK, width=2)
                for off in (-34, 34):
                    pts = oriented_rect(x - math.sin(a) * off, y + math.cos(a) * off, a, 190, 6)
                    c.poly(pts, fill=C("#ffe6a0") if not (i == 2 and off < 0) else COAT)
                c.poly(oriented_rect(x, y, a, 60, 40), fill=C("#9aa0a7"))
                gd.polygon([(px / 4, py / 4) for px, py in oriented_rect(x, y, a, 240, 130)], fill=(150, 110, 40))
            else:
                c.poly(oriented_rect(x, y, a, 210, 80), fill=C("#2a2f35"), outline=C("#0a0b0c"), width=2)
                c.poly(oriented_rect(x, y, a, 60, 40), fill=C("#353b42"))
                if i == 4:
                    for off in (-28, 28):
                        tx = x - math.cos(a) * 100 - math.sin(a) * off
                        ty = y - math.sin(a) * 100 + math.cos(a) * off
                        c.ell(tx, ty, 7, 7, fill=C("#ff2a2a"))
                        gd.ellipse([tx / 4 - 5, ty / 4 - 5, tx / 4 + 5, ty / 4 + 5], fill=(120, 0, 0))
    layer = c.im.reduce(SS)
    frame = frame.convert("RGBA")
    frame.alpha_composite(layer)
    img = ImageChops.add(frame.convert("RGB"), glow.filter(ImageFilter.GaussianBlur(6)).resize((W, H), BC))
    return camera(img, lerp(1.18, 1.0, ease(k / 3.5)), 540, lerp(1150, 960, ease(k / 3.5)))


def r_s8(rel, sc):
    t_abs = sc["start"] + rel
    lamp = "red" if t_abs >= E["s8_lock"] else "green"
    expr = "scared" if t_abs >= E["s8_lock"] + 0.3 else "dread"
    gaze = (0, 0) if t_abs >= E["s8_lock"] + 0.3 else (-1, 0)
    img = interior_shot(rel, expr, gaze, 0, 0, lamp, yard=True)
    k = t_abs - E["s8_dark"]
    if k >= 0:
        flick = [1, 0.2, 0.8, 0.1, 0.5][min(int(k / 0.1), 4)] if k < 0.5 else 0.0
        lvl = max(flick, 0.045)
        img = darken(img, lvl, (10, 15, 40), 0.4)
        if k > 0.5:
            fr = img.convert("RGBA")
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            a = int(230 * ease((k - 0.5) / 0.6))
            blink = 0 < (t_abs - E["title"] + 1.0) < 0.12
            for side in (-1, 1):
                ex, ey = 680 + side * 36.8, 792
                if blink:
                    d.line([ex - 18, ey, ex + 18, ey], fill=(230, 230, 230, a), width=3)
                else:
                    d.ellipse([ex - 18, ey - 14, ex + 18, ey + 14], fill=(235, 235, 228, a))
                    d.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=(10, 8, 8, a))
            fr.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.8)))
            img = fr.convert("RGB")
    return camera(img, lerp(1.0, 1.3, ease(rel / max(sc["dur"] - 1, 1))), 680, 860)


RENDER = dict(s1=r_s1, s2=r_s2, s3=r_s3, s4=r_s4, s5=r_s5, s6=r_s6, s7=r_s7, s8=r_s8)
CAPS = {}


def render_frame(t):
    sc = next((s for s in TL if s["start"] <= t < s["start"] + s["dur"]), TL[-1])
    img = RENDER[sc["id"]](t - sc["start"], sc).convert("RGBA")
    img.alpha_composite(A["vignette"])
    if t >= E["title"]:
        img = Image.blend(img.convert("RGB"), A["title"], ease((t - E["title"]) / 0.5)).convert("RGBA")
    else:
        lines = [ln for s in TL for ln in s["lines"]]
        for i, ln in enumerate(lines):
            nxt = lines[i + 1]["start"] if i + 1 < len(lines) else 1e9
            if ln["start"] - 0.05 <= t < min(ln["end"] + 0.35, nxt - 0.02):
                cap = CAPS.setdefault(ln["id"], caption_img(ln["cap"], ln["hl"]))
                comp(img, cap, 0, 1380 - cap.height / 2, clamp((t - ln["start"] + 0.05) / 0.1))
    return img.convert("RGB")


# --------------------------------------------------------------------------- main
TL, E = [], {}


def main():
    global TL, E
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="render QA stills only")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] narration (Windows SAPI scratch voice)...")
    raw = make_narration()
    durs = {k: len(v) / SR for k, v in raw.items()}
    TL = plan_timeline(durs)
    E = make_events(TL)

    print("\nTIMING (seconds)")
    for sc in TL:
        print(f"  {sc['id']}  {sc['start']:6.2f} -> {sc['start'] + sc['dur']:6.2f}  ({sc['dur']:.2f}s, voice x{sc['speed']:.2f})")
        for ln in sc["lines"]:
            print(f"        {ln['start']:6.2f}-{ln['end']:6.2f}  {ln['cap']}")
    print(f"  TOTAL {E['end']:.2f}s\n")
    (OUT_DIR / "timing.json").write_text(json.dumps(dict(scenes=TL, events=E), indent=2))

    print("[2/4] drawing layers -> assets/art ...")
    build_assets()

    if args.stills:
        sd = OUT_DIR / "stills"
        sd.mkdir(exist_ok=True)
        times = [0.5, 3.0]
        for sc in TL:
            times += [sc["start"] + sc["dur"] * f for f in (0.2, 0.55, 0.9)]
        times += [E["s5_flicks"][-1] + 0.3, E["s6_sign"] + 0.3, E["s8_dark"] + 1.0, E["end"] - 0.2]
        thumbs = []
        for t in sorted(times):
            im = render_frame(t)
            im.save(sd / f"t{t:05.2f}.png")
            thumbs.append(im.resize((270, 480), LZ))
        cols = 7
        sheet = Image.new("RGB", (cols * 270, math.ceil(len(thumbs) / cols) * 480), (0, 0, 0))
        for i, th in enumerate(thumbs):
            sheet.paste(th, ((i % cols) * 270, (i // cols) * 480))
        sheet.save(OUT_DIR / "contact_sheet.png")
        print(f"stills -> {sd}")
        return

    print("[3/5] audio mix...")
    stems, mix = build_audio(TL, E, raw)

    print(f"[4/5] rendering picture ({FPS} fps)...")
    ff = shutil.which("ffmpeg") or "ffmpeg"
    nframes = int(round(E["end"] * FPS))
    picture = OUT_DIR / f"picture_only_{FPS}fps.mp4"
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", str(picture)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(nframes):
        proc.stdin.write(render_frame(i / FPS).tobytes())
        if i % 120 == 0:
            print(f"  frame {i}/{nframes}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")

    print("[5/5] edit + final render...")
    sys.path.insert(0, str(Path(__file__).parent))
    from shotcut_project import find_melt, render_with_melt, write_mlt
    mlt = write_mlt(ROOT / "the_last_train_217.mlt", picture,
                    [("Narration", stems["narration_track.wav"], 0.0), ("SFX", stems["sfx_track.wav"], 0.0),
                     ("Music", stems["music_track.wav"], 0.0)], nframes, FPS, W, H)
    print(f"  Shotcut project -> {mlt}")
    out = OUT_DIR / f"the_last_train_217_{FPS}fps.mp4"
    edit = OUT_DIR / "_edit.mp4"
    melt = find_melt()
    if melt:
        print(f"  rendering with Shotcut's melt: {melt}")
        render_with_melt(melt, mlt, edit)
    else:
        print("  melt not found -> muxing with ffmpeg")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(picture), "-i", str(mix), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(edit)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(edit), "-c:v", "copy",
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", str(out)], check=True)
    edit.unlink()
    print(f"\nDONE -> {out}")


if __name__ == "__main__":
    main()
