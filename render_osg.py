"""Dark-orange "Omni OSG" re-edit of the speaker-shop talking-head reel.

Style follows the reference reel: condensed caps headlines (white line + orange
gradient line with glow) under a small letter-spaced label, numbered tag pills
with ticks, and B-roll pages on a charcoal board with an orange bloom, faint
floating play tiles and image cards with a glowing orange border.

The 0-3.33 s hook (incl. the colour-bar censor beep) is passed through untouched;
only the headline overlay is drawn on it. No vignette, no grade on the A-roll.

Usage: python3 render_osg.py SRC.mov IMG_DIR ASSET_DIR OUT.mp4   (env TEST=t1,t2 -> contact jpg)
"""
import os, subprocess, sys, math
import numpy as np, cv2, cairosvg, io
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, IMG, ASSET, OUT = sys.argv[1:5]
W, H, FPS = 1080, 1920, 30
DUR = 54.466
HOOK_END = 3.333

ORANGE = (255, 122, 38)
OR_TOP, OR_BOT = (255, 186, 92), (255, 96, 28)
RED = (255, 72, 60)
WHITE = (255, 255, 255)
LABEL = (236, 228, 218)

ANTON = f'{ASSET}/fonts/Anton.ttf'
INTER = f'{ASSET}/fonts/Inter.ttf'
_fc = {}
def anton(s):
    k = ('a', s)
    if k not in _fc: _fc[k] = ImageFont.truetype(ANTON, s)
    return _fc[k]
def inter(s, wght=650):
    k = ('i', s, wght)
    if k not in _fc:
        f = ImageFont.truetype(INTER, s)
        f.set_variation_by_axes([14, wght])
        _fc[k] = f
    return _fc[k]

def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3
def back(x):
    x = min(max(x, 0.0), 1.0)
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2

# ---------------------------------------------------------------- edit plan
# Text items: ('hdr', t_in, label, [(text, style, t_in), ...], y)  style: w=white, o=orange
#             ('big', t_in, text, style, y, size)                  chest punch word
#             ('tags', y, [(t_in, num, text, mark), ...])          mark: ok / no
#             ('emoji', t_in, code, x, y, size, bob)
#             ('pill', t_in, text, y)
# A shot: ('A', start, end, zoom, items)
# B shot: ('B', start, end, layout, card, items)   card = (img, crop, kb=(z0,z1,fx,fy), bg)
# V shot: ('V', start, end, items)  original footage, no zoom (hook / demo)
PLAN = [
    ('V', 0.0, HOOK_END, [
        ('hdr', 0.05, 'OMNI BEYOND  ·  OSG 6.5"', [('SPEAKER ADA BUNYI,', 'w', 0.15), ('TAPI FEEL TAK SAMPAI?', 'o', 1.35)], 120)]),
    ('A', HOOK_END, 5.96, 1.08, [
        ('hdr', 4.50, 'JUJURNYA', [('KEBANYAKAN SPEAKER', 'w', 4.55), ('SEKARANG NI HAMPEH!', 'o', 5.55)], 120)]),
    ('B', 5.96, 8.58, 'card', ('door', None, (1.04, 1.14, .5, .5), None), [
        ('hdr', 6.00, 'BILA BUKA', [('KUAT-KUAT SIKIT', 'w', 6.10)], 1010),
        ('tags', 1250, [(7.15, '01', 'BUNYI TENGGELAM', 'no'), (8.05, '02', 'BUNYI LARI', 'no')])]),
    ('A', 8.58, 9.35, 1.0, [
        ('big', 8.58, 'BETUL TAK?', 'o', 1080, 190)]),
    ('B', 9.35, 11.96, 'card', ('3', (150, 300, 1150, 900), (1.00, 1.08, .5, .5), (14, 12, 12)), [
        ('hdr', 9.40, 'OMNI BEYOND', [('SATU SET SPEAKER', 'w', 9.65), ('STEADY PUNYA', 'o', 11.05)], 1010)]),
    ('A', 11.96, 13.70, 1.10, [
        ('hdr', 12.00, 'SEBAB SPEAKER NI', [('RMS DIA', 'w', 12.95)], 120),
        ('big', 13.35, 'TINGGI', 'o', 1060, 260)]),
    ('A', 13.70, 15.45, 1.0, [
        ('hdr', 13.72, 'UNTUK YANG', [('SUKA DENGAR', 'w', 14.55), ('LAGU KUAT-KUAT', 'o', 14.95)], 120),
        ('emoji', 15.0, '1f50a', 880, 470, 120, True)]),
    ('B', 15.45, 18.62, 'card', ('3', (0, 330, 1206, 1180), (1.0, 1.30, .78, .8), None), [
        ('hdr', 15.48, 'SPESIFIKASI', [('OSG 6.5"', 'o', 15.55)], 1010),
        ('tags', 1250, [(16.10, '01', 'RMS 45 WATT', 'ok'), (17.20, '02', 'WATT TINGGI  ·  450W MAX', 'ok'),
                        (17.82, '03', 'MAGNET BESAR', 'ok')])]),
    ('A', 18.62, 22.04, 1.08, [
        ('hdr', 18.65, 'DENGAR LAGU', [], 120),
        ('tags', 170, [(18.95, '01', 'LAGU VOCAL  ·  LAYAN', 'ok'), (20.50, '02', 'LAGU BASS  ·  SEDAP', 'ok')])]),
    ('A', 22.04, 22.90, 1.0, [
        ('hdr', 22.06, 'JOM', [('DENGAR SENDIRI', 'o', 22.20)], 120),
        ('emoji', 22.3, '1f3b5', 900, 300, 110, True)]),
    ('V', 22.90, 33.93, [
        ('hdr', 23.20, 'DEMO BUNYI', [('OMNI BEYOND', 'w', 23.30), ('OSG 6.5"', 'o', 23.55)], 120),
        ('pill', 27.5, 'KUATKAN VOLUME', 1560)]),
    ('A', 33.93, 36.53, 1.08, [
        ('hdr', 33.95, 'MACAM MANA?', [], 120),
        ('big', 35.18, 'BUNYI', 'w', 1000, 150),
        ('big', 35.50, 'TAK PECAH!', 'o', 1150, 210)]),
    ('B', 36.53, 37.68, 'card', ('2', (0, 250, 589, 700), (1.0, 1.06, .5, .5), (238, 70, 70)), [
        ('hdr', 36.55, 'OMNI BEYOND', [('2 WAY + MID BASS', 'o', 36.62)], 1010),
        ('tags', 1250, [(36.75, '01', 'OSG 6.5  ·  2 WAY', 'ok'), (37.25, '02', 'OSG 6.5B  ·  MID BASS', 'ok')])]),
    ('A', 37.68, 40.16, 1.0, [
        ('hdr', 37.70, 'DEKAT BEG KUNING', [('ADA SATU SET', 'w', 39.40), ('SPECIAL!', 'o', 39.72)], 120)]),
    ('B', 40.16, 41.94, 'card', ('1', None, (1.0, 1.10, .45, .5), (255, 255, 255)), [
        ('hdr', 40.18, 'KAKAK BUAT', [('SET', 'w', 40.25), ('PLUG & PLAY', 'o', 40.75)], 1010),
        ('tags', 1400, [(41.20, '01', 'SOKET SIAP DISEDIAKAN', 'ok')])]),
    ('A', 41.94, 42.98, 1.10, [
        ('hdr', 41.96, 'PERGI TENGOK', [('BEG KUNING', 'o', 42.40)], 120),
        ('emoji', 42.5, '1f447', 150, 1250, 150, True)]),
    ('B', 42.98, 46.70, 'card', ('3', (150, 300, 1150, 900), (1.08, 1.0, .5, .5), (14, 12, 12)), [
        ('hdr', 43.00, 'SELALU SPEC MACAM NI', [('HARGA', 'w', 45.00)], 1010),
        ('price', 45.55, 'RM500', 46.25)]),
    ('A', 46.70, 47.62, 1.12, [
        ('big', 46.74, 'TAK PAYAH!', 'o', 1080, 220)]),
    ('B', 47.62, 49.30, 'card', ('4', (0, 150, 600, 450), (1.0, 1.05, .5, .5), (255, 255, 255)), [
        ('hdr', 47.64, 'KAKAK BELANJA', [('HARGA PALING', 'w', 47.70), ('RAHMAH', 'o', 48.55)], 1010)]),
    ('B', 49.30, 50.87, 'phone', ('2', None, (1.0, 1.04, .5, .62), None), [
        ('hdr', 49.32, 'DEKAT BEG KUNING', [('BAWAH RM300', 'o', 49.62)], 110),
        ('circle', 49.95, (20, 765, 360, 805))]),
    ('A', 50.87, 52.95, 1.08, [
        ('hdr', 50.90, 'SO NAK TAHU', [('HARGA BERAPA', 'w', 51.28), ('NI KAK?', 'o', 52.38)], 120)]),
    ('B', 52.95, DUR, 'phone', ('2', None, (1.04, 1.0, .5, .5), None), [
        ('hdr', 52.97, 'KLIK', [('BEG KUNING', 'o', 53.05)], 110),
        ('emoji', 53.4, '1f447', 880, 250, 130, True)]),
]

# ---------------------------------------------------------------- primitives
def grad_fill(w, h, top, bot):
    a = np.linspace(0, 1, h)[:, None, None]
    g = (np.array(top)[None, None] * (1 - a) + np.array(bot)[None, None] * a)
    return np.repeat(g, w, axis=1).astype(np.uint8)

_tc = {}
def text_rgba(txt, fnt, style, glow=True, shadow=True, track=0):
    """RGBA layer for one line, padded for glow/shadow. Returns (layer, pad)."""
    key = (txt, id(fnt), style, glow, shadow, track)
    if key in _tc: return _tc[key]
    pad = 40
    if track:
        widths = [fnt.getlength(c) + track for c in txt]
        tw = int(sum(widths))
    else:
        tw = int(fnt.getlength(txt))
    asc, desc = fnt.getmetrics()
    th = asc + desc
    m = Image.new('L', (tw + 2 * pad, th + 2 * pad), 0)
    d = ImageDraw.Draw(m)
    if track:
        x = pad
        for c, w_ in zip(txt, widths):
            d.text((x, pad), c, font=fnt, fill=255); x += w_
    else:
        d.text((pad, pad), txt, font=fnt, fill=255)
    mw, mh = m.size
    if style == 'o':
        col = grad_fill(mw, mh, OR_TOP, OR_BOT)
        # gradient over the glyph box only
        col[pad:pad + th] = grad_fill(mw, th, OR_TOP, OR_BOT)
    elif style == 'r':
        col = np.zeros((mh, mw, 3), np.uint8); col[:] = RED
    elif style == 'l':
        col = np.zeros((mh, mw, 3), np.uint8); col[:] = LABEL
    else:
        col = np.zeros((mh, mw, 3), np.uint8); col[:] = WHITE
    fg = np.dstack([col, np.array(m)]).astype(np.float32) / 255
    out = np.zeros_like(fg)
    if shadow:
        sh = np.array(m.filter(ImageFilter.GaussianBlur(9))).astype(np.float32) / 255
        sh = np.roll(sh, 4, axis=0) * 0.62
        out[..., 3] = sh
    if glow and style == 'o':
        gl = np.array(m.filter(ImageFilter.GaussianBlur(16))).astype(np.float32) / 255 * 0.75
        ga = gl
        oa = out[..., 3]
        na = ga + oa * (1 - ga)
        out[..., :3] = (np.array(ORANGE) / 255 * ga[..., None] + out[..., :3] * (oa * (1 - ga))[..., None]) / np.maximum(na, 1e-6)[..., None]
        out[..., 3] = na
    # glyph over
    fa = fg[..., 3]
    oa = out[..., 3]
    na = fa + oa * (1 - fa)
    out[..., :3] = (fg[..., :3] * fa[..., None] + out[..., :3] * (oa * (1 - fa))[..., None]) / np.maximum(na, 1e-6)[..., None]
    out[..., 3] = na
    _tc[key] = (out, pad, tw, th)
    return _tc[key]

def comp(frame, rgba, x, y, alpha=1.0, clip=None):
    """Alpha-composite float RGBA (0-1) onto uint8 frame at x,y; clip=(y0,y1) in frame coords."""
    h, w = rgba.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if clip:
        y0, y1 = max(y0, clip[0]), min(y1, clip[1])
    if x1 <= x0 or y1 <= y0 or alpha <= 0: return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4] * alpha
    dst = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (src[..., :3] * 255 * a + dst * (1 - a)).astype(np.uint8)

def line_in(frame, t, t0, txt, fnt, style, x, y, shadow=True, track=0):
    """Mask-rise reveal: line slides up from beneath its own baseline box."""
    if t < t0: return 0
    lay, pad, tw, th = text_rgba(txt, fnt, style, shadow=shadow, track=track)
    e = ease((t - t0) / 0.28)
    dy = int((1 - e) * th * 0.95)
    comp(frame, lay, x - pad, y - pad + dy, alpha=min(1, e * 1.6), clip=(y - pad, y + th + 6))
    return th

def rrect_mask(w, h, r):
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    return m

# ---------------------------------------------------------------- items
def draw_hdr(frame, t, it, a_roll):
    _, t0, label, lines, y = it
    if t < t0: return
    x = 64
    lab = text_rgba(label, inter(25, 650), 'l', glow=False, shadow=a_roll, track=7)
    e = ease((t - t0) / 0.25)
    comp(frame, lab[0], x - lab[1], y - lab[1], alpha=e)
    yy = y + 44
    for txt, st, ti in lines:
        size = 124 if len(txt) <= 16 else (110 if len(txt) <= 20 else 96)
        th = line_in(frame, t, ti, txt, anton(size), st, x, yy, shadow=a_roll)
        yy += int(anton(size).getmetrics()[0] * 1.02) if th or t < ti else 0

def draw_big(frame, t, it):
    _, t0, txt, st, y, size = it
    if t < t0: return
    f = anton(size)
    lay, pad, tw, th = text_rgba(txt, f, st)
    while tw > W - 120:
        size -= 10; f = anton(size); lay, pad, tw, th = text_rgba(txt, f, st)
    k = back((t - t0) / 0.3)
    s = 0.7 + 0.3 * k
    if abs(s - 1) > 0.01:
        lay = cv2.resize(lay, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    h, w = lay.shape[:2]
    comp(frame, lay, (W - w) // 2, y + th // 2 - h // 2, alpha=min(1, (t - t0) / 0.12))

_tagc = {}
def tag_rgba(num, txt, mark):
    key = (num, txt, mark)
    if key in _tagc: return _tagc[key]
    f = anton(54); fn = inter(24, 800)
    tw = int(f.getlength(txt))
    w, h = tw + 230, 96
    pad = 30
    im = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    glow = Image.new('L', im.size, 0)
    ImageDraw.Draw(glow).rounded_rectangle((pad, pad, pad + w, pad + h), 18, outline=255, width=6)
    glow = glow.filter(ImageFilter.GaussianBlur(10))
    gl = Image.new('RGBA', im.size, ORANGE + (0,)); gl.putalpha(glow.point(lambda v: int(v * 0.7)))
    im = Image.alpha_composite(im, gl)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((pad, pad, pad + w, pad + h), 18, fill=(34, 29, 28, 228), outline=ORANGE + (255,), width=3)
    d.text((pad + 26, pad + h // 2), num, font=fn, fill=ORANGE, anchor='lm')
    d.text((pad + 76, pad + h // 2 + 2), txt, font=f, fill=WHITE, anchor='lm')
    cx = pad + w - 52; cy = pad + h // 2
    if mark is None:
        pass
    elif mark == 'ok':
        d.line([(cx - 17, cy + 1), (cx - 4, cy + 15), (cx + 20, cy - 17)], fill=ORANGE, width=6, joint='curve')
    else:
        d.line([(cx - 14, cy - 14), (cx + 14, cy + 14)], fill=RED, width=6)
        d.line([(cx - 14, cy + 14), (cx + 14, cy - 14)], fill=RED, width=6)
    arr = np.array(im).astype(np.float32) / 255
    _tagc[key] = (arr, pad)
    return _tagc[key]

def draw_tags(frame, t, it):
    _, y, rows = it
    for i, (t0, num, txt, mark) in enumerate(rows):
        if t < t0: continue
        arr, pad = tag_rgba(num, txt, mark)
        e = ease((t - t0) / 0.25)
        comp(frame, arr, 64 - pad - int((1 - e) * 60), y + i * 122 - pad, alpha=e)

_emo = {}
def emoji(code, size):
    k = (code, size)
    if k not in _emo:
        png = cairosvg.svg2png(url=f'{ASSET}/tw/{code}.svg', output_width=size, output_height=size)
        im = Image.open(io.BytesIO(png)).convert('RGBA')
        pad = 20
        base = Image.new('RGBA', (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
        sh = Image.new('L', base.size, 0); sh.paste(im.split()[3], (pad, pad + 6))
        sh = sh.filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.5))
        shl = Image.new('RGBA', base.size, (0, 0, 0, 0)); shl.putalpha(sh)
        base = Image.alpha_composite(base, shl); base.alpha_composite(im, (pad, pad))
        _emo[k] = np.array(base).astype(np.float32) / 255
    return _emo[k]

def draw_emoji(frame, t, it):
    _, t0, code, x, y, size, bob = it
    if t < t0: return
    k = back((t - t0) / 0.35)
    s = max(4, int(size * (0.4 + 0.6 * k)))
    arr = emoji(code, size)
    if s != size:
        arr = cv2.resize(arr, (s + 40 * s // size, s + 40 * s // size))
    dy = int(10 * math.sin(4.0 * (t - t0))) if bob else 0
    h, w = arr.shape[:2]
    comp(frame, arr, x - w // 2, y - h // 2 + dy, alpha=min(1, (t - t0) / 0.1))

def draw_pill(frame, t, it):
    _, t0, txt, y = it
    if t < t0: return
    arr, pad = tag_rgba('', txt, None)
    # reuse tag look but centred, with speaker icon instead of number
    e = ease((t - t0) / 0.3)
    h, w = arr.shape[:2]
    x = (W - w) // 2
    comp(frame, arr, x, y + int((1 - e) * 40), alpha=e)
    ic = emoji('1f50a', 40)
    comp(frame, ic, x + pad + 6, y + int((1 - e) * 40) + pad + 48 - ic.shape[0] // 2, alpha=e)

def draw_price(frame, t, it):
    _, t0, txt, t_strike = it
    if t < t0: return
    f = anton(330)
    lay, pad, tw, th = text_rgba(txt, f, 'w')
    k = back((t - t0) / 0.3)
    s = 0.6 + 0.4 * k
    l2 = cv2.resize(lay, None, fx=s, fy=s) if abs(s - 1) > 0.01 else lay
    h, w = l2.shape[:2]
    cx, cy = 64 + tw // 2, 1320 + th // 2
    comp(frame, l2, cx - w // 2, cy - h // 2, alpha=min(1, (t - t0) / 0.1))
    if t >= t_strike:
        e = ease((t - t_strike) / 0.2)
        x0, x1 = 50, 50 + int((tw + 40) * e)
        cv2.line(frame, (x0, cy + 30), (x1, cy - 30), RED, 22, cv2.LINE_AA)

def draw_circle(frame, t, it, card_xform):
    _, t0, box = it
    if t < t0: return
    e = ease((t - t0) / 0.4)
    x0, y0, x1, y1 = card_xform(box)
    c = ((x0 + x1) // 2, (y0 + y1) // 2); ax = ((x1 - x0) // 2 + 30, (y1 - y0) // 2 + 26)
    glow = np.zeros_like(frame)
    cv2.ellipse(glow, c, ax, -4, -200, -200 + 370 * e, ORANGE, 14, cv2.LINE_AA)
    g = cv2.GaussianBlur(glow, (0, 0), 8)
    np.copyto(frame, np.clip(frame.astype(np.int16) + (g * 0.8).astype(np.int16), 0, 255).astype(np.uint8))
    cv2.ellipse(frame, c, ax, -4, -200, -200 + 370 * e, ORANGE, 7, cv2.LINE_AA)

# ---------------------------------------------------------------- B-roll page
def make_bg():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32); base[:] = (20, 17, 20)
    d = np.sqrt(((x - 540) / 1.0) ** 2 + ((y - 1250) / 1.15) ** 2)
    g = np.clip(1 - d / 980, 0, 1) ** 1.8
    base += g[..., None] * np.array([150, 52, 14], np.float32)
    d2 = np.sqrt((x - 900) ** 2 + (y - 250) ** 2)
    base += (np.clip(1 - d2 / 700, 0, 1) ** 2)[..., None] * np.array([60, 20, 14], np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)
BG = make_bg()
TILES = [(110, 150, -8), (420, 70, 6), (760, 110, -5), (980, 260, 9), (60, 1680, 7), (330, 1780, -6),
         (640, 1700, 5), (920, 1620, -9), (200, 1500, 4), (1010, 1840, 3)]
def tile_rgba():
    w, h, pad = 150, 100, 10
    im = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((pad, pad, pad + w, pad + h), 14, fill=(70, 62, 62, 120))
    cx, cy = pad + w // 2, pad + h // 2
    d.polygon([(cx - 12, cy - 15), (cx - 12, cy + 15), (cx + 16, cy)], fill=(120, 110, 108, 150))
    d.rounded_rectangle((pad + 12, pad + 12, pad + 50, pad + 18), 3, fill=(110, 100, 98, 120))
    return im
TILE = tile_rgba()
_tile_rot = {a: np.array(TILE.rotate(a, expand=True, resample=Image.BICUBIC)).astype(np.float32) / 255 for _, _, a in TILES}

_img = {}
def load(name):
    if name not in _img:
        p = f'{ASSET}/web/door.jpg' if name == 'door' else f'{IMG}/{name}.jpg'
        _img[name] = np.array(Image.open(p).convert('RGB'))
    return _img[name]

def fit_cover(im, cw, ch, z, fx, fy):
    h, w = im.shape[:2]
    s = max(cw / w, ch / h) * z
    sw, sh = cw / s, ch / s
    x0 = min(max(fx * w - sw / 2, 0), w - sw); y0 = min(max(fy * h - sh / 2, 0), h - sh)
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(im, M, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), (s, x0, y0)

def fit_contain(im, cw, ch, z, fx, fy, bg):
    h, w = im.shape[:2]
    s = min(cw / w, ch / h) * z
    out = np.zeros((ch, cw, 3), np.uint8); out[:] = bg
    r = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    rh, rw = r.shape[:2]
    ox, oy = (cw - rw) // 2, (ch - rh) // 2
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    ww, hh = min(rw - sx0, cw - dx0), min(rh - sy0, ch - dy0)
    out[dy0:dy0 + hh, dx0:dx0 + ww] = r[sy0:sy0 + hh, sx0:sx0 + ww]
    return out, (s, -ox / s, -oy / s)

_glow = {}
def card_glow(w, h):
    k = (w, h)
    if k not in _glow:
        pad = 50
        m = Image.new('L', (w + 2 * pad, h + 2 * pad), 0)
        ImageDraw.Draw(m).rounded_rectangle((pad, pad, pad + w, pad + h), 26, outline=255, width=10)
        m = np.array(m.filter(ImageFilter.GaussianBlur(18))).astype(np.float32) / 255 * 0.85
        lay = np.zeros(m.shape + (4,), np.float32); lay[..., :3] = np.array(ORANGE) / 255; lay[..., 3] = m
        # drop shadow
        sh = Image.new('L', (w + 2 * pad, h + 2 * pad), 0)
        ImageDraw.Draw(sh).rounded_rectangle((pad, pad + 18, pad + w, pad + h + 18), 26, fill=255)
        sh = np.array(sh.filter(ImageFilter.GaussianBlur(22))).astype(np.float32) / 255 * 0.6
        out = np.zeros_like(lay); out[..., 3] = sh
        a = lay[..., 3]; na = a + sh * (1 - a)
        out[..., :3] = lay[..., :3] * (a / np.maximum(na, 1e-6))[..., None]; out[..., 3] = na
        _glow[k] = (out, pad, np.array(rrect_mask(w, h, 24)).astype(np.float32)[..., None] / 255)
    return _glow[k]

def b_frame(shot, t):
    _, t0, t1, layout, card, items = shot
    frame = BG.copy()
    # floating tiles
    for (tx, ty, a) in TILES:
        arr = _tile_rot[a]
        comp(frame, arr, tx - arr.shape[1] // 2, int(ty - arr.shape[0] // 2 + 8 * math.sin(0.9 * t + tx)), alpha=0.9)
    name, crop, kb, bgc = card
    im = load(name)
    if crop: im = im[crop[1]:crop[3], crop[0]:crop[2]]
    if layout == 'card':
        cx, cy, cw, ch = 70, 300, 940, 640
    else:
        cw = 640; ch = int(cw * im.shape[0] / im.shape[1]); ch = min(ch, 1360)
        cx, cy = (W - cw) // 2, 520
    p = (t - t0) / max(t1 - t0, 0.01)
    z = kb[0] + (kb[1] - kb[0]) * p
    if bgc is None:
        pic, xf = fit_cover(im, cw, ch, z, kb[2], kb[3])
    else:
        pic, xf = fit_contain(im, cw - 40, ch - 40, z, kb[2], kb[3], bgc)
        pad_ = np.zeros((ch, cw, 3), np.uint8); pad_[:] = bgc; pad_[20:-20, 20:-20] = pic; pic = pad_
        xf = (xf[0], xf[1] - 20 / xf[0], xf[2] - 20 / xf[0])
    e = ease((t - t0) / 0.32)
    off = int((1 - e) * 70)
    sc = 0.94 + 0.06 * e
    glow, gp, mask = card_glow(cw, ch)
    card_rgba = np.dstack([pic.astype(np.float32) / 255, mask[..., 0]])
    # orange border
    bd = Image.new('L', (cw, ch), 0); ImageDraw.Draw(bd).rounded_rectangle((0, 0, cw - 1, ch - 1), 24, outline=255, width=4)
    bd = np.array(bd).astype(np.float32)[..., None] / 255
    card_rgba[..., :3] = card_rgba[..., :3] * (1 - bd) + np.array(OR_TOP) / 255 * bd
    if sc != 1:
        card_rgba = cv2.resize(card_rgba, None, fx=sc, fy=sc); glow_s = cv2.resize(glow, None, fx=sc, fy=sc)
    else:
        glow_s = glow
    ox = cx + (cw - card_rgba.shape[1]) // 2; oy = cy + off + (ch - card_rgba.shape[0]) // 2
    comp(frame, glow_s, ox - int(gp * sc), oy - int(gp * sc), alpha=e)
    comp(frame, card_rgba, ox, oy, alpha=e)
    def card_xform(box):
        s, x0, y0 = xf
        return [int(cx + (box[0] - x0) * s), int(cy + (box[1] - y0) * s), int(cx + (box[2] - x0) * s), int(cy + (box[3] - y0) * s)]
    for it in items:
        draw_item(frame, t, it, False, card_xform)
    return frame

def draw_item(frame, t, it, a_roll, card_xform=None):
    k = it[0]
    if k == 'hdr': draw_hdr(frame, t, it, a_roll)
    elif k == 'big': draw_big(frame, t, it)
    elif k == 'tags': draw_tags(frame, t, it)
    elif k == 'emoji': draw_emoji(frame, t, it)
    elif k == 'pill': draw_pill(frame, t, it)
    elif k == 'price': draw_price(frame, t, it)
    elif k == 'circle': draw_circle(frame, t, it, card_xform)

# ---------------------------------------------------------------- A-roll
# Gentle global grade for the talking-head footage: soft highlight shoulder, a touch
# less exposure and saturation. Uniform over the frame (no vignette, no edge darkening).
def _curve():
    x = np.arange(256, dtype=np.float32)
    y = x * 0.93
    k = 120.0
    hi = x > k
    y[hi] = k * 0.93 + (x[hi] - k) * 0.72
    # tiny S in the mids for a bit of depth without crushing the shadows
    y = y + 4.0 * np.sin(np.pi * np.clip((x - 40) / 170, 0, 1)) * np.sign(x - 125) * (np.abs(x - 125) < 85)
    return np.clip(y, 0, 255).astype(np.uint8)
CURVE = _curve()
SAT = 0.80
BARS = (2.68, 3.34)   # colour-bar censor card in the hook stays untouched

def grade(src, t):
    if BARS[0] <= t < BARS[1]: return src
    f = cv2.LUT(src, CURVE).astype(np.float32)
    lum = f @ np.float32([0.299, 0.587, 0.114])
    f = lum[..., None] + (f - lum[..., None]) * SAT
    return np.clip(f, 0, 255).astype(np.uint8)

def zoom(src, z, t, t0, cy=820):
    # short ease into the new framing on each cut (punch-in feel), centred on the speaker
    zz = z + 0.035 * (1 - ease((t - t0) / 0.22)) if z > 1.0 else z
    if zz <= 1.0001: return src
    M = np.float32([[zz, 0, W / 2 * (1 - zz)], [0, zz, cy * (1 - zz)]])
    return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def shot_at(t):
    for s in PLAN:
        if s[1] <= t < s[2]: return s
    return PLAN[-1]

def render(src, t):
    s = shot_at(t)
    if s[0] == 'B':
        return b_frame(s, t)
    src = grade(src, t)
    if s[0] == 'A':
        frame = zoom(src, s[3], t, s[1]).copy(); items = s[4]
    else:
        frame = src.copy(); items = s[3]
    for it in items:
        draw_item(frame, t, it, True)
    return frame

def decoder(start=0.0):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', SRC, '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-'], stdout=subprocess.PIPE)

FB = W * H * 3
if os.environ.get('TEST'):
    ts = [float(x) for x in os.environ['TEST'].split(',')]
    tiles = []
    for t in ts:
        p = decoder(t); raw = p.stdout.read(FB); p.kill()
        f = render(np.frombuffer(raw, np.uint8).reshape(H, W, 3), t)
        tiles.append(cv2.resize(f, (360, 640)))
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    Image.fromarray(np.vstack(rows)).save(OUT)
    sys.exit()

enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                        '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
                        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', OUT], stdin=subprocess.PIPE)
dec = decoder()
n = int(round(DUR * FPS))
for i in range(n):
    raw = dec.stdout.read(FB)
    if len(raw) < FB: break
    t = i / FPS
    enc.stdin.write(render(np.frombuffer(raw, np.uint8).reshape(H, W, 3), t).tobytes())
    if i % 150 == 0: print(f'{t:.1f}s', flush=True)
enc.stdin.close(); enc.wait()
