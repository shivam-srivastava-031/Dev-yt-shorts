#!/usr/bin/env python3
"""
The Last Train at 2:17 AM -- 2.5D (depth / parallax) version.

Every shot is split into depth layers (far background, mid-ground, characters,
blurred foreground, floating dust, fog). A virtual pinhole camera trucks, dollies and
orbits through them, so near layers move faster than far ones -- the "3D" feel --
while the art stays 2D. Re-uses the story timing and audio stems of the flat build.

Run build/make_short.py once first (it creates the narration, stems and timing.json).

Usage:
    python build/depth25.py            # -> output/the_last_train_217_2.5D_24fps.mp4
    python build/depth25.py --stills   # QA stills -> output/stills_2.5D/
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
import make_short as ms  # noqa: E402
from make_short import (BOLD, C, Cv, FPS, H, INK, W, clamp, clip_to, comp, ease, ease_out, font,  # noqa: E402
                        lerp, with_alpha)
from shotcut_project import find_melt, render_with_melt, write_mlt  # noqa: E402

BL = Image.Resampling.BILINEAR
BC = Image.Resampling.BICUBIC
F = 1000.0          # focal length (world units)
B = {}              # 2.5D-only assets


# --------------------------------------------------------------------------- camera
class Cam:
    """Pinhole camera. Layers are authored to look 1:1 at rest; D = layer distance.
    dx/dy: camera translation (parallax), zc: dolly forward, fx/fy: framing point (pan),
    zl: lens zoom (>1 gives overscan), df: focus distance kept steady while orbiting."""

    def __init__(self, dx=0.0, dy=0.0, zc=0.0, fx=W / 2, fy=H / 2, zl=1.08, df=1000.0, orbit=True):
        self.dx, self.dy, self.zc, self.zl, self.df, self.orbit = dx, dy, zc, zl, df, orbit
        mx, my = W / (2 * zl), H / (2 * zl)
        self.fx, self.fy = clamp(fx, mx, W - mx), clamp(fy, my, H - my)

    def scale(self, D):
        return self.zl * D / (D - self.zc)

    def offset(self, D):
        k = 1 / (D - self.zc) - (1 / (self.df - self.zc) if self.orbit else 0)
        return -F * self.dx * k * self.zl, -F * self.dy * k * self.zl

    def pt(self, x, y, D):
        s = self.scale(D)
        ox, oy = self.offset(D)
        return W / 2 + (x - self.fx) * s + ox, H / 2 + (y - self.fy) * s + oy


def xform(img, x, y, D, cam):
    s = cam.scale(D)
    ox, oy = cam.offset(D)
    a = 1 / s
    c = cam.fx - x - (W / 2 + ox) / s
    f = cam.fy - y - (H / 2 + oy) / s
    return img.transform((W, H), Image.Transform.AFFINE, (a, 0, c, 0, a, f), resample=BL)


def put(frame, img, x, y, D, cam, alpha=1.0):
    if D - cam.zc < 30 or alpha <= 0.001:
        return
    s = cam.scale(D)
    if img.width * img.height * s * s > 0.3 * W * H:
        out = xform(img, x, y, D, cam)
        frame.alpha_composite(with_alpha(out, alpha) if alpha < 0.999 else out)
    else:
        sx, sy = cam.pt(x, y, D)
        if sx > W or sy > H or sx + img.width * s < 0 or sy + img.height * s < 0:
            return
        im2 = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), BL)
        comp(frame, im2, sx, sy, alpha)


# --------------------------------------------------------------------------- assets
def rgba(img):
    return img.convert("RGBA")


def soft_dot(n=64, col=(255, 236, 200)):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n / 2 + .5, yy - n / 2 + .5) / (n / 2)
    arr = np.zeros((n, n, 4), np.uint8)
    arr[..., :3] = col
    arr[..., 3] = (np.clip(1 - r, 0, 1) ** 2 * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def gradient(top, bottom, w=W, h=H):
    yy = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(top, np.float32) * (1 - yy) + np.array(bottom, np.float32) * yy
    arr = np.repeat(arr, w, axis=1)
    out = np.dstack([arr, np.full((h, w, 1), 255, np.float32)]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def fog_layer(seed, cell, alpha, col=(150, 160, 178)):
    rng = np.random.default_rng(seed)
    pad = 400
    small = (rng.random(((H + 2 * pad) // cell + 2, (W + 2 * pad) // cell + 2)) * 255).astype(np.uint8)
    img = Image.fromarray(small, "L").resize((W + 2 * pad, H + 2 * pad), BC).filter(ImageFilter.GaussianBlur(cell / 3))
    a = np.clip((np.asarray(img, np.float32) / 255 - .5) * 3.2, 0, 1) * alpha
    arr = np.zeros((H + 2 * pad, W + 2 * pad, 4), np.uint8)
    arr[..., :3] = col
    arr[..., 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA"), pad


def strap():
    c = Cv(120, 460)
    c.rect(48, 0, 72, 330, fill=C("#3c4045"), outline=INK, width=2, r=6)
    c.ell(60, 380, 44, 52, outline=C("#e8d9a8"), width=14)
    return c.out().filter(ImageFilter.GaussianBlur(5))


def pole(w=34, h=H + 400, blur=0):
    c = Cv(w + 8, h)
    c.rect(4, 0, w + 4, h, fill=C("#c9ced2"), outline=C("#6f767c"), width=3)
    c.rect(4 + w * .35, 0, 4 + w * .55, h, fill=C("#f1f4f6"))
    img = c.out()
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


def pillar():
    c = Cv(260, H + 300)
    c.rect(0, 0, 260, H + 300, fill=C("#8f8466"), outline=INK, width=4)
    for y in range(0, H + 300, 46):
        c.line([(0, y), (260, y)], C("#776d53"), 3)
    c.rect(200, 0, 260, H + 300, fill=C("#6e664f"))
    return c.out().filter(ImageFilter.GaussianBlur(9))


def clock_sprite():
    c = Cv(300, 440)
    for x in (80, 220):
        c.rect(x - 6, 0, x + 6, 270, fill=C("#2b2e33"))
    c.rect(10, 262, 290, 382, fill=C("#0b0b0b"), outline=C("#3a3d42"), width=6, r=12)
    c.text(132, 322, "2:17", ms.MONO, 92, C("#ffb020"))
    c.text(250, 346, "AM", BOLD, 26, C("#ffb020"))
    c.rect(50, 388, 250, 424, fill=C("#20252b"), r=6)
    c.text(150, 406, "LAST TRAIN", BOLD, 22, C("#f1f1f1"))
    img = c.out()
    g = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rectangle([30, 285, 230, 360], fill=(255, 150, 20, 90))
    g = g.filter(ImageFilter.GaussianBlur(16))
    g.alpha_composite(img)
    return g


def light_beams():
    c = Cv(W, 1500)
    for x in (180, 540, 900):
        c.poly([(x - 110, 98), (x + 110, 98), (x + 330, 1450), (x - 330, 1450)], fill=C("#ffe6aa", 26))
    return c.out().filter(ImageFilter.GaussianBlur(30))


def extruded(text, size, face, side_a, side_b, depth=18):
    f = font(size, BOLD)
    im = Image.new("RGBA", (W, size + 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cy = (size + 120) / 2 - 10
    for i in range(depth, 0, -1):
        t = i / depth
        col = tuple(int(lerp(side_a[j], side_b[j], t)) for j in range(3)) + (255,)
        d.text((W / 2 + i * 0.9, cy + i * 1.3), text, font=f, anchor="mm", fill=col)
    d.text((W / 2, cy), text, font=f, anchor="mm", fill=face)
    return im


def build_depth_assets():
    A = ms.A
    plat = rgba(ms.bg_platform(clock=False))
    B["p_wall"] = plat.crop((0, 0, W, 790)).filter(ImageFilter.GaussianBlur(1.2))
    B["p_farpit"] = plat.crop((0, 685, W, 1015))
    B["p_nearpit"] = plat.crop((0, 985, W, 1455))
    B["p_floor"] = plat.crop((0, 1440, W, H))
    B["p_clock"] = clock_sprite()
    B["p_beams"] = light_beams()
    B["p_pillar"] = pillar()

    wall = rgba(ms.bg_interior(poles=False))
    wall.putalpha(ImageChops.invert(A["int_mask"]))
    B["int_wall"] = wall
    B["int_far"] = gradient((8, 14, 26), (16, 26, 40))
    B["int_mask"] = A["int_mask"]
    B["pole"] = pole()
    B["strap"] = strap()
    sh = ms.maya_seated("tired").getchannel("A")
    shadow = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    shadow.putalpha(sh.point(lambda v: int(v * .5)))
    B["maya_shadow"] = shadow.filter(ImageFilter.GaussianBlur(16))

    ww = rgba(A["window"])
    ww.putalpha(ImageChops.invert(A["glass_mask"]))
    B["win_wall"] = ww
    B["glass_mask"] = A["glass_mask"]
    B["tunnel"] = gradient((11, 19, 32), (21, 33, 52))
    B["fg_pole"] = pole(70, H + 400, blur=10)

    B["tilt_bg"], _ = ms.tilt_composite(with_maya=False)
    c = Cv(W, 1250)
    ms.draw_maya_closeup(c, dy=1800)
    B["maya_close"] = c.out()

    B["fog1"], B["fog_pad"] = fog_layer(11, 220, 55, (205, 212, 225))
    B["fog2"], _ = fog_layer(23, 300, 45, (170, 180, 200))

    B["title_bg"] = gradient((14, 18, 30), (3, 3, 6))
    B["t1"] = extruded("THE LAST TRAIN", 86, (242, 242, 242, 255), (60, 60, 70), (20, 20, 26))
    B["t2"] = extruded("AT 2:17 AM", 112, (255, 176, 32, 255), (150, 80, 0), (50, 20, 0))
    rule = Image.new("RGBA", (480, 8), (192, 57, 43, 255))
    B["t_rule"] = rule

    B["dot"] = soft_dot()
    rng = np.random.default_rng(5)
    B["dust"] = [(rng.uniform(-300, W + 300), rng.uniform(-300, H + 300), rng.uniform(480, 3200),
                  rng.uniform(2.5, 7), rng.uniform(0, 6.28)) for _ in range(80)]
    B["dots"] = {}

    for name in ("p_wall", "p_clock", "p_pillar", "int_wall", "win_wall", "maya_close", "fog1", "t1", "t2"):
        B[name].save(ms.ART_DIR / f"depth_{name}.png")


def draw_dust(frame, cam, t, amount=1.0):
    for x, y, D, sz, ph in B["dust"]:
        if D - cam.zc < 120:
            continue
        xx = x + 14 * t + 25 * math.sin(t * .4 + ph)
        yy = y + 18 * math.sin(t * .3 + ph * 1.7) - 6 * t
        sx, sy = cam.pt(xx, yy, D)
        r = sz * cam.scale(D) * (1.8 if D < 800 else 1.0)
        if not (-60 < sx < W + 60 and -60 < sy < H + 60):
            continue
        a = amount * (0.18 if D < 800 else 0.38) * (0.55 + 0.45 * math.sin(t * 1.3 + ph * 3))
        n = max(2, int(r * 2))
        dot = B["dots"].get(n)
        if dot is None:
            dot = B["dots"][n] = B["dot"].resize((n, n), BL)
        comp(frame, dot, sx - n / 2, sy - n / 2, a)


# --------------------------------------------------------------------------- shots
def window_comp(t, cam, maya_expr="uneasy", man=None, man_alpha=1.0, man_scale=0.72, man_pos=(450, 600),
                phone=False, other_alpha=0.22, drift=(0, 0), streak=1.0, refl_scale=1.0):
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    content = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    put(content, B["tunnel"], 0, 0, 4000, cam)
    d = ImageDraw.Draw(content)
    for i in range(9):
        y = 300 + i * 118 + (i * 37) % 60
        x = 1100 - ((t * 1900 * streak + i * 530) % 2600)
        D = 2500 + (i % 3) * 400
        x0, y0 = cam.pt(x, y, D)
        x1, y1 = cam.pt(x + 160 + (i % 3) * 90, y + 6, D)
        d.rounded_rectangle([x0, y0, max(x1, x0 + 4), max(y1, y0 + 3)], radius=3, fill=(255, 220, 160, 70))
    if man:
        sx, sy = math.sin(t * 2.3) * 8, math.sin(t * 3.1 + 1) * 5
        mx, my = man_pos[0] + drift[0] + sx, man_pos[1] + drift[1] + sy
        put(content, B_otherwin(), 420 + drift[0] + sx - 360, 700 + drift[1] + sy - 300, 2100, cam, other_alpha)
        img = ms.scaled(ms.stranger(man, phone), man_scale)
        put(content, img, mx - 300 * man_scale, my - 282 * man_scale, 1900, cam, man_alpha)
        if phone:
            gl = ms.scaled(ms.A["phone_glow"], man_scale)
            put(content, gl, mx + 5 * man_scale - gl.width / 2, my + 268 * man_scale - gl.height / 2, 1880, cam, man_alpha)
    if maya_expr:
        r = ms.scaled(ms.maya_reflection(maya_expr), refl_scale)
        put(content, r, 610 - 300 * refl_scale, 880 - 230 * refl_scale, 1500, cam)
    clip_to(content, xform(B["glass_mask"], 0, 0, 1000, cam))
    frame.alpha_composite(content)
    put(frame, B["win_wall"], 0, 0, 1000, cam)
    put(frame, ms.A["glare"], 0, 0, 990, cam)
    put(frame, B["fg_pole"], 900, -200, 520, cam)
    put(frame, B["strap"], 120, -40, 640, cam)
    draw_dust(frame, cam, t)
    return frame.convert("RGB")


def B_otherwin():
    return ms.A["otherwin"]


def interior_comp(t, cam, expr="tired", gaze=(0, 0), turn=0.0, tilt=0.0, lamp="green", yard=False):
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    content = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    put(content, B["int_far"], 0, 0, 3000, cam)
    d = ImageDraw.Draw(content)
    if yard:
        for i in range(3):
            x = 1200 - ((t * 260 + i * 700) % 2100)
            x0, y0 = cam.pt(x, 520 + i * 90, 2600)
            x1, y1 = cam.pt(x + 220, 640 + i * 90, 2600)
            d.ellipse([x0, y0, x1, y1], fill=(255, 150, 40, 90))
        content = content.filter(ImageFilter.GaussianBlur(22))
    else:
        for i in range(6):
            x = 1100 - ((t * 2100 + i * 610) % 2600)
            x0, y0 = cam.pt(x, 430 + i * 80, 2600)
            x1, y1 = cam.pt(x + 180, 437 + i * 80, 2600)
            d.rounded_rectangle([x0, y0, max(x1, x0 + 4), max(y1, y0 + 3)], radius=3, fill=(255, 230, 170, 110))
    clip_to(content, xform(B["int_mask"], 0, 0, 1300, cam))
    frame.alpha_composite(content)
    put(frame, B["int_wall"], 0, 0, 1300, cam)
    col = {"green": (60, 220, 110), "red": (255, 50, 40)}[lamp]
    lx, ly = cam.pt(540, 600, 1300)
    s = cam.scale(1300)
    dd = ImageDraw.Draw(frame)
    g = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse([20, 20, 100, 100], fill=col + (120,))
    gs = g.filter(ImageFilter.GaussianBlur(14)).resize((int(120 * s), int(120 * s)), BL)
    comp(frame, gs, lx - 60 * s, ly - 60 * s)
    dd.ellipse([lx - 16 * s, ly - 16 * s, lx + 16 * s, ly + 16 * s], fill=col + (255,))
    put(frame, B["maya_shadow"], 380 + 26, 550 + 20, 1290, cam)
    put(frame, ms.maya_seated(expr, gaze, turn, tilt), 380, 550, 1150, cam)
    put(frame, B["pole"], 110 - 21, -200, 950, cam)
    put(frame, B["pole"], 1012 - 21, -200, 950, cam)
    for x in (230, 560, 880):
        put(frame, B["strap"], x - 60, -60, 660, cam)
    draw_dust(frame, cam, t, 0.8)
    return frame.convert("RGB")


def d_s1(rel, sc):
    p = rel / sc["dur"]
    cam = Cam(dx=lerp(-70, 70, p), dy=8 * math.sin(rel * .9), zc=lerp(0, 230, ease(p)), fx=520, fy=760, df=1600)
    img = window_comp(rel, cam, "uneasy", "calm")
    if 0 <= sc["start"] + rel - ms.E["s1_flicker"] < 0.1:
        img = ms.darken(img, 0.15)
    return img


def d_s2(rel, sc):
    E = ms.E
    p = rel / sc["dur"]
    cam = Cam(dx=lerp(100, -100, p), dy=6 * math.sin(rel * .7), zc=lerp(0, 150, ease(p)), fx=540, fy=1000, df=1000)
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    put(frame, B["p_wall"], 0, 0, 2600, cam)
    put(frame, B["p_farpit"], 0, 685, 2300, cam)
    xf = lerp(1150, -400, ease_out((rel - 0.25) / 2.9))
    xn = lerp(1150, -530, ease_out(rel / 2.8))
    put(frame, ms.A["far_train"], xf, 690, 2250, cam)
    put(frame, B["p_beams"], 0, 0, 2200, cam)
    put(frame, B["p_clock"], 390, 0, 2000, cam)
    put(frame, B["p_nearpit"], 0, 985, 1600, cam)
    t_abs = sc["start"] + rel
    o = 100 * ease((t_abs - E["s2_open"]) / 0.45) * (1 - ease((t_abs - E["s2_close"]) / 0.45))
    near = ms.A["near_train"].copy()
    dp = ms.A["door_panel"]
    if o < 109:
        near.alpha_composite(dp, (980, 26), (int(o), 0, 110, dp.height))
        near.alpha_composite(dp.crop((0, 0, max(1, int(110 - o)), dp.height)), (int(1090 + o), 26))
    put(frame, near, xn, 1000, 1550, cam)
    put(frame, B["p_floor"], 0, 1440, 1150, cam)
    q = clamp((t_abs - E["s2_walk0"]) / (E["s2_walk1"] - E["s2_walk0"]))
    if q < 1:
        x, y = lerp(800, 560, ease(q)), lerp(1770, 1460, ease(q))
        s = lerp(1.0, 0.78, ease(q))
        moving = 0 < q < 1
        step = int(rel * 4) % 2 if moving else 0
        bob = abs(math.sin(rel * 2 * math.pi * 2)) * 10 if moving else 0
        m = ms.scaled(ms.maya_standing(step), s)
        put(frame, m, x - m.width / 2, y - m.height + 10 - bob, lerp(1000, 1450, ease(q)), cam,
            1 - ease((q - 0.82) / 0.18))
    put(frame, B["p_pillar"], -150, -150, 520, cam)
    draw_dust(frame, cam, rel)
    if rel < 2.4:
        lab = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
        ImageDraw.Draw(lab).text((W / 2, 60), "10 MINUTES EARLIER", font=font(46, BOLD), anchor="mm",
                                 fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0, 255))
        comp(frame, lab, 0, 560, min(1, rel / 0.3) * (1 - ease((rel - 1.9) / 0.5)))
    return frame.convert("RGB")


def d_s3(rel, sc):
    t_abs = sc["start"] + rel
    if t_abs < ms.E["s3_cut"]:
        tilt = -4 + 3 * math.sin(rel * 1.7)
        cam = Cam(dx=lerp(-80, 30, ease(rel / 4)), dy=5 * math.sin(rel * .8), zc=lerp(0, 190, ease(rel / 4)),
                  fx=600 + 8 * math.sin(rel * 1.3), fy=980, df=1150)
        return interior_comp(rel, cam, "tired", (0, 0.4), 0, tilt)
    k = t_abs - ms.E["s3_cut"]
    cam = Cam(dx=lerp(50, -40, k / 3.5), zc=lerp(60, 200, ease(k / 3.5)), fx=620, fy=820, df=1500)
    return window_comp(rel, cam, "tired")


def d_s4(rel, sc):
    E = ms.E
    t_abs = sc["start"] + rel
    if t_abs < E["s4_turn"]:
        a = ease((t_abs - E["s4_reveal"]) / 0.6)
        expr = "scared" if t_abs > E["s4_reveal"] + 0.7 else "tired"
        k = ease(rel / 5)
        cam = Cam(dx=lerp(10, -60, k), dy=4 * math.sin(rel), zc=lerp(80, 270, k), fx=lerp(620, 500, k),
                  fy=lerp(820, 700, k), df=1800)
        return window_comp(rel, cam, expr, "calm", a)
    p = ease((t_abs - E["s4_pan"]) / 0.7)
    cam = Cam(dx=25 * math.sin(rel * .9), dy=0, zc=lerp(0, 60, p), zl=lerp(1.25, 1.3, p), fx=lerp(680, 300, p),
              fy=lerp(900, 1120, p), df=1150)
    return interior_comp(rel, cam, "scared", (-1, 0), -1.0, 3)


def d_s5(rel, sc):
    t_abs = sc["start"] + rel
    n, black = ms.s5_state(t_abs)
    scale = [0.72, 0.88, 1.02, 1.16, 1.26][n] + (0.05 * ease((rel - 2) / 4) if n >= 4 else 0)
    pos = [(450, 600), (455, 600), (460, 600), (465, 600), (470, 600)][n]
    expr = "urgent" if n >= 2 else "calm"
    cam = Cam(dx=6 * math.sin(rel * 23) + 4 * math.sin(rel * 37) + lerp(-30, 30, rel / 7),
              dy=5 * math.sin(rel * 29), zc=lerp(120, 260, ease(rel / 7)), fx=540, fy=760, df=1800)
    img = window_comp(rel, cam, "scared", expr, 1.0, scale, pos, phone=n >= 4, refl_scale=0.92)
    lvl = 0.08 if black else 0.85 + 0.15 * math.sin(rel * 37) * math.sin(rel * 13)
    return ms.darken(img, lvl)


def d_s6(rel, sc):
    E = ms.E
    t_abs = sc["start"] + rel
    if t_abs < E["s6_tilt"]:
        k = ease(rel / 4)
        cam = Cam(dx=lerp(50, -30, k), dy=5 * math.sin(rel * .8), zc=lerp(0, 230, k), fx=540, fy=900, df=1000)
        frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        put(frame, B["tunnel"], 0, 0, 4000, cam)
        d = ImageDraw.Draw(frame)
        for i in range(9):
            y = 200 + i * 190
            x = 1100 - ((rel * 1500 + i * 530) % 2600)
            x0, y0 = cam.pt(x, y, 2600)
            x1, y1 = cam.pt(x + 200, y + 8, 2600)
            d.rounded_rectangle([x0, y0, max(x1, x0 + 4), max(y1, y0 + 3)], radius=3, fill=(255, 220, 160, 60))
        ph = ms.A["phone"].rotate(-3, resample=BC, center=(540, 960))
        put(frame, ph, 0, 0, 1000, cam)
        if int(rel * 2.4) % 2 == 0:
            x0, y0, x1, y1 = ms.A["cursor"]
            p0, p1 = cam.pt(x0, y0, 1000), cam.pt(x1, y1, 1000)
            ImageDraw.Draw(frame).rectangle([p0[0], p0[1], p1[0], p1[1]], fill=(20, 120, 255, 255))
        put(frame, ms.A["glare"], 0, 0, 960, cam, 0.8)
        draw_dust(frame, cam, rel, 0.7)
        return frame.convert("RGB")
    p = ease((t_abs - E["s6_tilt"]) / 1.6)
    y0 = lerp(ms.TILT_H - H, 0, p)
    zl = 1.05
    bg = B["tilt_bg"].crop((0, int(y0), W, int(y0) + H)).convert("RGBA")
    frame = bg.resize((int(W * zl), int(H * zl)), BL).crop((int(W * (zl - 1) / 2), int(H * (zl - 1) / 2),
                                                               int(W * (zl - 1) / 2) + W, int(H * (zl - 1) / 2) + H))
    my = 1800 - y0 * 1.0 + 600 * p
    mz = 1.08 + 0.1 * p
    m = B["maya_close"].resize((int(W * mz), int(B["maya_close"].height * mz)), BL)
    comp(frame, m, (W - m.width) / 2, H / 2 + (my - H / 2) * mz)
    cam = Cam(dy=lerp(-40, 40, p), zl=1.0, orbit=False)
    draw_dust(frame, cam, rel, 0.8)
    img = frame.convert("RGB")
    if t_abs >= E["s6_sign"]:
        k = t_abs - E["s6_sign"]
        x, y, w, h = ms.A["led_box"]
        g = Image.new("RGB", (W, H), (0, 0, 0))
        cx = W / 2
        ImageDraw.Draw(g).rectangle([cx + (x - cx) * zl, H / 2 + (y - y0 - H / 2) * zl,
                                     cx + (x + w - cx) * zl, H / 2 + (y + h - y0 - H / 2) * zl], fill=(255, 40, 30))
        g = g.filter(ImageFilter.GaussianBlur(40))
        from PIL import ImageEnhance
        g = ImageEnhance.Brightness(g).enhance(0.6 + 0.6 * math.exp(-k * 3) + 0.15 * (int(k * 4) % 2))
        img = ImageChops.add(img, g)
        shake = math.exp(-k * 5) * 18
        img = ms.camera(img, 1.04 + 0.04 * ease(k / 2), 540 + shake * math.sin(k * 60), 600 + shake * math.cos(k * 47))
    return img


def d_s7(rel, sc):
    E = ms.E
    t_abs = sc["start"] + rel
    if t_abs < E["s7_aerial"]:
        span = E["s7_aerial"] - sc["start"]
        p = ease_out((rel - 0.25) / (span - 0.25))
        drift = (-950 * p ** 1.3, -140 * p)
        cam = Cam(dx=lerp(-50, 50, ease(rel / span)), zc=160, fx=540, fy=760, df=1900)
        return window_comp(rel, cam, "dread", "urgent", 1.0, 1.1, (470, 600), other_alpha=lerp(0.3, 0.75, ease(rel / 0.8)),
                           drift=drift, streak=0.7)
    k = t_abs - E["s7_aerial"]
    q = ease_out(k / 3.6)
    cam = Cam(dx=lerp(-60, 60, q), zc=lerp(700, 0, q), fx=540, fy=lerp(1150, 960, q), df=3000)
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    put(frame, stage_aerial(k), 0, 0, 3000, cam)
    pad = B["fog_pad"]
    put(frame, B["fog2"], -pad + 30 * k, -pad - 20 * k, 1900, cam)
    put(frame, B["fog1"], -pad - 60 * k, -pad + 25 * k, 1200, cam)
    return frame.convert("RGB")


def stage_aerial(k):
    """Ground, tracks, trains and train glow, un-cameraed (flat build's aerial, no camera move)."""
    frame = ms.A["aerial"].copy()
    c = Cv(W, H)
    glow = Image.new("RGB", (W // 4, H // 4), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for path, lit in ((ms.PATH_L, True), (ms.PATH_R, False)):
        head = 620 + 330 * k + (0 if lit else -40)
        for i in range(5):
            s = head - i * 222 - 105
            x, y, a = ms.path_at(path, s)
            if lit:
                c.poly(ms.oriented_rect(x, y, a, 210, 80), fill=C("#c7ccd2"), outline=INK, width=2)
                for off in (-34, 34):
                    pts = ms.oriented_rect(x - math.sin(a) * off, y + math.cos(a) * off, a, 190, 6)
                    c.poly(pts, fill=C("#ffe6a0") if not (i == 2 and off < 0) else ms.COAT)
                c.poly(ms.oriented_rect(x, y, a, 60, 40), fill=C("#9aa0a7"))
                gd.polygon([(px / 4, py / 4) for px, py in ms.oriented_rect(x, y, a, 240, 130)], fill=(150, 110, 40))
            else:
                c.poly(ms.oriented_rect(x, y, a, 210, 80), fill=C("#2a2f35"), outline=C("#0a0b0c"), width=2)
                c.poly(ms.oriented_rect(x, y, a, 60, 40), fill=C("#353b42"))
                if i == 4:
                    for off in (-28, 28):
                        tx = x - math.cos(a) * 100 - math.sin(a) * off
                        ty = y - math.sin(a) * 100 + math.cos(a) * off
                        c.ell(tx, ty, 7, 7, fill=C("#ff2a2a"))
                        gd.ellipse([tx / 4 - 5, ty / 4 - 5, tx / 4 + 5, ty / 4 + 5], fill=(120, 0, 0))
    frame = frame.convert("RGBA")
    frame.alpha_composite(c.im.reduce(ms.SS))
    img = ImageChops.add(frame.convert("RGB"), glow.filter(ImageFilter.GaussianBlur(6)).resize((W, H), BC))
    return img.convert("RGBA")


def d_s8(rel, sc):
    E = ms.E
    t_abs = sc["start"] + rel
    lamp = "red" if t_abs >= E["s8_lock"] else "green"
    expr = "scared" if t_abs >= E["s8_lock"] + 0.3 else "dread"
    gaze = (0, 0) if t_abs >= E["s8_lock"] + 0.3 else (-1, 0)
    p = ease(rel / max(sc["dur"] - 1, 1))
    cam = Cam(dx=lerp(60, -30, p), dy=4 * math.sin(rel * .7), zc=lerp(0, 330, p), fx=680, fy=860, df=1150)
    img = interior_comp(rel, cam, expr, gaze, 0, 0, lamp, yard=True)
    k = t_abs - E["s8_dark"]
    if k >= 0:
        flick = [1, 0.2, 0.8, 0.1, 0.5][min(int(k / 0.1), 4)] if k < 0.5 else 0.0
        img = ms.darken(img, max(flick, 0.045), (10, 15, 40), 0.4)
        if k > 0.5:
            fr = img.convert("RGBA")
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            a = int(230 * ease((k - 0.5) / 0.6))
            blink = 0 < (t_abs - E["title"] + 1.0) < 0.12
            s = cam.scale(1150)
            for side in (-1, 1):
                ex, ey = cam.pt(680 + side * 36.8, 792, 1150)
                if blink:
                    d.line([ex - 18 * s, ey, ex + 18 * s, ey], fill=(230, 230, 230, a), width=max(2, int(3 * s)))
                else:
                    d.ellipse([ex - 18 * s, ey - 14 * s, ex + 18 * s, ey + 14 * s], fill=(235, 235, 228, a))
                    d.ellipse([ex - 6 * s, ey - 6 * s, ex + 6 * s, ey + 6 * s], fill=(10, 8, 8, a))
            fr.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.8)))
            img = fr.convert("RGB")
    return img


def d_title(k):
    q = ease(k / 2.0)
    cam = Cam(dx=lerp(60, -60, q), dy=lerp(-20, 20, q), zc=lerp(0, 160, q), fx=540, fy=930, df=1150)
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    put(frame, B["title_bg"], 0, 0, 3000, cam)
    draw_dust(frame, cam, k + 50, 0.9)
    put(frame, B["t1"], 0, 840 - B["t1"].height / 2 + 10, 1350, cam)
    put(frame, B["t_rule"], 300, 912, 1200, cam)
    put(frame, B["t2"], 0, 1010 - B["t2"].height / 2 + 10, 1000, cam)
    return frame.convert("RGB")


RENDER = dict(s1=d_s1, s2=d_s2, s3=d_s3, s4=d_s4, s5=d_s5, s6=d_s6, s7=d_s7, s8=d_s8)


def render_frame(t):
    TL, E = ms.TL, ms.E
    sc = next((s for s in TL if s["start"] <= t < s["start"] + s["dur"]), TL[-1])
    img = RENDER[sc["id"]](t - sc["start"], sc).convert("RGBA")
    img.alpha_composite(ms.A["vignette"])
    if t >= E["title"]:
        k = t - E["title"]
        title = d_title(k).convert("RGBA")
        title.alpha_composite(ms.A["vignette"])
        return Image.blend(img.convert("RGB"), title.convert("RGB"), ease(k / 0.5))
    lines = [ln for s in TL for ln in s["lines"]]
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else 1e9
        if ln["start"] - 0.05 <= t < min(ln["end"] + 0.35, nxt - 0.02):
            cap = ms.CAPS.setdefault(ln["id"], ms.caption_img(ln["cap"], ln["hl"]))
            comp(img, cap, 0, 1380 - cap.height / 2, clamp((t - ln["start"] + 0.05) / 0.1))
    return img.convert("RGB")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    args = ap.parse_args()
    tj = ms.OUT_DIR / "timing.json"
    stems = {n: ms.AUDIO_DIR / n for n in ("narration_track.wav", "sfx_track.wav", "music_track.wav")}
    if not tj.exists() or not all(p.exists() for p in stems.values()):
        sys.exit("Run build/make_short.py first (it creates timing.json and the audio stems).")
    data = json.loads(tj.read_text())
    ms.TL, ms.E = data["scenes"], data["events"]
    E = ms.E
    print("drawing layers...")
    ms.build_assets()
    build_depth_assets()

    if args.stills:
        sd = ms.OUT_DIR / "stills_2.5D"
        sd.mkdir(exist_ok=True)
        times = [0.3, 2.5, 5.0]
        for sc in ms.TL:
            times += [sc["start"] + sc["dur"] * f for f in (0.25, 0.6, 0.92)]
        times += [E["s6_sign"] + 0.3, E["s7_aerial"] + 0.3, E["s7_aerial"] + 2.5, E["end"] - 0.6]
        thumbs = []
        for t in sorted(times):
            im = render_frame(t)
            im.save(sd / f"t{t:05.2f}.png")
            thumbs.append(im.resize((270, 480), ms.LZ))
        cols = 7
        sheet = Image.new("RGB", (cols * 270, math.ceil(len(thumbs) / cols) * 480))
        for i, th in enumerate(thumbs):
            sheet.paste(th, ((i % cols) * 270, (i // cols) * 480))
        sheet.save(ms.OUT_DIR / "contact_sheet_2.5D.png")
        print("stills ->", sd)
        return

    ff = shutil.which("ffmpeg") or "ffmpeg"
    nframes = int(round(E["end"] * FPS))
    picture = ms.OUT_DIR / f"picture_only_2.5D_{FPS}fps.mp4"
    print(f"rendering {nframes} frames at {FPS} fps...")
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                             "-pix_fmt", "yuv420p", str(picture)], stdin=subprocess.PIPE)
    for i in range(nframes):
        proc.stdin.write(render_frame(i / FPS).tobytes())
        if i % 120 == 0:
            print(f"  frame {i}/{nframes}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")

    mlt = write_mlt(ms.ROOT / "the_last_train_217_2.5D.mlt", picture,
                    [("Narration", stems["narration_track.wav"], 0.0), ("SFX", stems["sfx_track.wav"], 0.0),
                     ("Music", stems["music_track.wav"], 0.0)], nframes, FPS, W, H)
    print("Shotcut project ->", mlt)
    edit = ms.OUT_DIR / "_edit_2.5D.mp4"
    out = ms.OUT_DIR / f"the_last_train_217_2.5D_{FPS}fps.mp4"
    melt = find_melt()
    if melt:
        print("rendering with Shotcut's melt:", melt)
        render_with_melt(melt, mlt, edit)
    else:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(picture), "-i", str(ms.AUDIO_DIR / "full_mix.wav"),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(edit)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(edit), "-c:v", "copy", "-af",
                    "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", str(out)], check=True)
    edit.unlink()
    print("\nDONE ->", out)


if __name__ == "__main__":
    main()
