"""Writes a Shotcut/MLT project (V1 + narration/SFX/music tracks) and renders it with Shotcut's melt."""
import os
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape


def find_melt():
    cands = [shutil.which("melt")]
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
        if base:
            cands += [str(Path(base) / "Shotcut" / "melt.exe"), str(Path(base) / "Programs" / "Shotcut" / "melt.exe")]
    return next((c for c in cands if c and Path(c).exists()), None)


def _producer(pid, path, nframes, extra=""):
    return (f'  <producer id="{pid}" in="0" out="{nframes - 1}">\n'
            f'    <property name="length">{nframes}</property>\n'
            f'    <property name="resource">{escape(Path(path).as_posix())}</property>\n'
            f'    <property name="mlt_service">avformat</property>\n{extra}'
            f'  </producer>\n')


def write_mlt(out_mlt, video, stems, nframes, fps, w, h):
    """stems: list of (track_name, wav_path, gain_db)."""
    x = ['<?xml version="1.0" encoding="utf-8"?>\n',
         '<mlt LC_NUMERIC="C" version="7.0.0" title="The Last Train at 2:17 AM" producer="main_bin">\n',
         f'  <profile description="Vertical 1080x1920 {fps}fps" width="{w}" height="{h}" progressive="1" '
         f'sample_aspect_num="1" sample_aspect_den="1" display_aspect_num="9" display_aspect_den="16" '
         f'frame_rate_num="{fps}" frame_rate_den="1" colorspace="709"/>\n',
         '  <playlist id="main_bin">\n    <property name="xml_retain">1</property>\n  </playlist>\n',
         f'  <producer id="black" in="0" out="{nframes - 1}">\n    <property name="length">{nframes}</property>\n'
         '    <property name="resource">0</property>\n    <property name="mlt_service">color</property>\n'
         '    <property name="mlt_image_format">rgba</property>\n  </producer>\n',
         f'  <playlist id="background">\n    <entry producer="black" in="0" out="{nframes - 1}"/>\n  </playlist>\n',
         _producer("vid", video, nframes),
         f'  <playlist id="playlist0">\n    <property name="shotcut:video">1</property>\n'
         f'    <property name="shotcut:name">V1</property>\n    <entry producer="vid" in="0" out="{nframes - 1}"/>\n'
         '  </playlist>\n']
    for i, (name, path, gain_db) in enumerate(stems, start=1):
        flt = (f'    <filter id="vol{i}">\n      <property name="window">75</property>\n'
               f'      <property name="max_gain">20dB</property>\n      <property name="level">{gain_db:.1f}</property>\n'
               '      <property name="mlt_service">volume</property>\n'
               '      <property name="shotcut:filter">audioGain</property>\n    </filter>\n')
        x.append(_producer(f"aud{i}", path, nframes, flt))
        x.append(f'  <playlist id="playlist{i}">\n    <property name="shotcut:audio">1</property>\n'
                 f'    <property name="shotcut:name">A{i} {escape(name)}</property>\n'
                 f'    <entry producer="aud{i}" in="0" out="{nframes - 1}"/>\n  </playlist>\n')
    x.append(f'  <tractor id="tractor0" title="The Last Train at 2:17 AM" in="0" out="{nframes - 1}">\n'
             '    <property name="shotcut">1</property>\n'
             '    <track producer="background"/>\n    <track producer="playlist0"/>\n')
    for i in range(1, len(stems) + 1):
        x.append(f'    <track producer="playlist{i}" hide="video"/>\n')
    x.append('    <transition id="vblend">\n      <property name="a_track">0</property>\n'
             '      <property name="b_track">1</property>\n      <property name="mlt_service">qtblend</property>\n'
             '      <property name="always_active">1</property>\n    </transition>\n')
    for i in range(1, len(stems) + 2):
        x.append(f'    <transition id="mix{i}">\n      <property name="a_track">0</property>\n'
                 f'      <property name="b_track">{i}</property>\n      <property name="mlt_service">mix</property>\n'
                 '      <property name="always_active">1</property>\n      <property name="sum">1</property>\n'
                 '    </transition>\n')
    x.append('  </tractor>\n</mlt>\n')
    Path(out_mlt).write_text("".join(x), encoding="utf-8")
    return out_mlt


def render_with_melt(melt, mlt, out_mp4):
    cmd = [melt, "-silent", str(mlt), "-consumer", f"avformat:{out_mp4}", "vcodec=libx264", "crf=18", "preset=medium",
           "pix_fmt=yuv420p", "acodec=aac", "ab=192k", "ar=48000", "channels=2", "movflags=+faststart", "real_time=-1"]
    subprocess.run(cmd, check=True)
    return out_mp4
