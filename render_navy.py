"""'Navy CapCut' re-edit of the subwoofer/gain talking-head reel.

Matches the reference reel: deep-navy motion-graphic pages (dashed arcs, sparkle
stars, gold swoosh, glass icon tiles, mouse cursor), white Poppins captions with a
big orange-gradient Montserrat Black Italic keyword, and A-roll overlays that sit
*behind* the speaker (person cutout via MediaPipe selfie segmentation).

Usage: python3 render_navy.py SRC.mov OUT.mp4 [SFX.json]   (env TEST=t1,t2 -> contact jpg)
"""
import os, sys, json, math, subprocess, io
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cairosvg
from ai_edge_litert.interpreter import Interpreter

SRC, OUT = sys.argv[1], sys.argv[2]
SFX_OUT = sys.argv[3] if len(sys.argv) > 3 else None
HERE = os.path.dirname(os.path.abspath(__file__))
BR = f'{HERE}/assets/broll'
FD = f'{HERE}/assets/fonts'
TW = os.environ.get('TWEMOJI', '/tmp/twemoji/package')
SEG_MODEL = os.environ.get('SEG_MODEL', 'selfie_segmentation.tflite')
W, H, FPS = 1080, 1920, 30

WHITE = (255, 255, 255)
ORANGE_TOP, ORANGE_BOT = (255, 214, 102), (255, 118, 24)
GOLD = (233, 182, 82)
NAVY_C, NAVY_E = np.array([22, 60, 138], np.float32), np.array([3, 11, 34], np.float32)

FP6, FP6I, FP7 = 'poppins-latin-600-normal.ttf', 'poppins-latin-600-italic.ttf', 'poppins-latin-700-normal.ttf'
FM9I, FM8 = 'montserrat-latin-900-italic.ttf', 'montserrat-latin-800-normal.ttf'
_fonts = {}
def font(name, size):
    k = (name, size)
    if k not in _fonts: _fonts[k] = ImageFont.truetype(f'{FD}/{name}', size)
    return _fonts[k]

SFX = set()
def sfx(kind, t):
    SFX.add((kind, round(t, 2)))


# ---------------------------------------------------------------- maths
def clip01(x): return min(max(x, 0.0), 1.0)
def ease(x): x = clip01(x); return 1 - (1 - x) ** 3
def back(x):
    x = clip01(x); c1 = 1.9; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def prog(t, t0, d): return clip01((t - t0) / d)


# ---------------------------------------------------------------- compositing
def put(fr, L, cx, cy, a=1.0, s=1.0, rot=0.0):
    """alpha-composite RGBA float layer L centred at (cx, cy), scaled and rotated"""
    if a <= 0.003 or s <= 0.02: return
    h, w = L.shape[:2]
    if s != 1.0 or rot:
        r = math.radians(rot)
        nw = int(abs(w * s * math.cos(r)) + abs(h * s * math.sin(r))) + 2
        nh = int(abs(w * s * math.sin(r)) + abs(h * s * math.cos(r))) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, s)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        L = cv2.warpAffine(L, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        h, w = nh, nw
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    sub = L[y0 - y:y1 - y, x0 - x:x1 - x]
    al = sub[..., 3:4] * (a / 255.0)
    reg = fr[y0:y1, x0:x1]
    if reg.shape[2] == 4:  # compositing into another RGBA layer
        ra = reg[..., 3:4] / 255
        na = al + ra * (1 - al)
        reg[..., :3] = (sub[..., :3] * al + reg[..., :3] * ra * (1 - al)) / np.maximum(na, 1e-4)
        reg[..., 3:4] = na * 255
        return
    reg += (sub[..., :3] - reg) * al


def pil2f(im): return np.asarray(im.convert('RGBA')).astype(np.float32)


# ---------------------------------------------------------------- text
_tc = {}
def T(text, fname, size, fill=WHITE, glow=None, shadow=0.6, track=0, maxw=980):
    """text layer: solid colour or 'orange' vertical gradient, soft shadow, optional glow"""
    key = (text, fname, size, fill, glow, shadow, track, maxw)
    if key in _tc: return _tc[key]
    f = font(fname, size)
    def width(f): return sum(f.getlength(c) + track for c in text) - track if track else f.getlength(text)
    while f.size > 28 and width(f) > maxw: f = font(fname, int(f.size * 0.94))
    asc, desc = f.getmetrics()
    pad = int(f.size * 0.45) + 10
    tw = int(width(f)) + 2 * pad; th = asc + desc + 2 * pad
    m = Image.new('L', (tw, th), 0); d = ImageDraw.Draw(m)
    if track:
        x = pad
        for c in text: d.text((x, pad), c, font=f, fill=255); x += f.getlength(c) + track
    else:
        d.text((pad, pad), text, font=f, fill=255)
    if fill == 'orange':
        g = np.linspace(0, 1, th)[:, None, None]
        top, bot = np.array(ORANGE_TOP, np.float32), np.array(ORANGE_BOT, np.float32)
        lo, hi = pad + asc * 0.15, pad + asc + desc * 0.2
        g = np.clip((np.arange(th) - lo) / (hi - lo), 0, 1)[:, None, None]
        col = np.repeat(top * (1 - g) + bot * g, tw, axis=1)
    else:
        col = np.ones((th, tw, 3), np.float32) * np.array(fill, np.float32)
    ma = np.asarray(m).astype(np.float32)
    out = np.zeros((th, tw, 4), np.float32)
    if shadow:
        sh = np.asarray(m.filter(ImageFilter.GaussianBlur(size * 0.09 + 3))).astype(np.float32)
        sh = np.roll(sh, int(size * 0.05) + 2, axis=0)
        out[..., 3] = sh * shadow
    if glow:
        gl = np.asarray(m.filter(ImageFilter.GaussianBlur(size * 0.22))).astype(np.float32) * glow
        gc = np.array(ORANGE_BOT if fill == 'orange' else (120, 170, 255), np.float32)
        a0 = out[..., 3:4] / 255; a1 = np.clip(gl[..., None] * 1.6, 0, 255) / 255
        out[..., :3] = out[..., :3] * a0 * (1 - a1) + gc * a1
        out[..., 3:4] = (a0 + a1 * (1 - a0)) * 255
        out[..., :3] /= np.maximum(out[..., 3:4] / 255, 1e-4)
    a0 = out[..., 3:4] / 255; a1 = ma[..., None] / 255
    na = a1 + a0 * (1 - a1)
    out[..., :3] = (col * a1 + out[..., :3] * a0 * (1 - a1)) / np.maximum(na, 1e-4)
    out[..., 3:4] = na * 255
    _tc[key] = out
    return out

def ink_w(L, size): return L.shape[1] - 2 * (int(size * 0.45) + 10)


def caption(fr, t, t0, lead, key, tail, ytop, tail_at=None, ksize=126):
    """reference caption: white words pop in one by one, then a big orange italic keyword"""
    y = ytop
    i = 0
    if lead:
        words = lead.split()
        lays = [T(w_, FP6, 54) for w_ in words]
        sp = 16
        lines, cur, cw = [], [], 0
        for L in lays:
            iw = ink_w(L, 54)
            if cur and cw + sp + iw > 940: lines.append((cur, cw)); cur, cw = [], 0
            cw = cw + (sp if cur else 0) + iw; cur.append(L)
        lines.append((cur, cw))
        for ln, lw in lines:
            x = W / 2 - lw / 2
            for L in ln:
                iw = ink_w(L, 54)
                p = ease((t - t0 - i * 0.075) / 0.18)
                if p > 0: put(fr, L, x + iw / 2, y + 36 + (1 - p) * 14, p)
                x += iw + sp; i += 1
            y += 70
        y += 8
    tk = t0 + i * 0.075 + (0.04 if lead else 0)
    if t >= tk: sfx('pop', tk)
    if key:
        L = T(key, FM9I, ksize, 'orange', glow=0.55, maxw=1000)
        p = prog(t, tk, 0.28)
        if p > 0: put(fr, L, W / 2, y + ksize * 0.55, min(1, p * 2.5), 0.55 + 0.45 * back(p))
        y += ksize * 1.08
    if tail:
        tt = t0 + tail_at if tail_at else tk + 0.3
        L = T(tail, FP6I, 46, maxw=960)
        p = ease((t - tt) / 0.22)
        if p > 0: put(fr, L, W / 2, y + 30 + (1 - p) * 12, p)


# ---------------------------------------------------------------- icons / images
_ic = {}
def emoji(cp, size):
    k = (cp, size)
    if k not in _ic:
        png = cairosvg.svg2png(url=f'{TW}/{cp}.svg', output_width=size, output_height=size)
        im = Image.open(io.BytesIO(png)).convert('RGBA')
        sh = Image.new('RGBA', (size + 60, size + 60), (0, 0, 0, 0))
        a = Image.new('L', sh.size, 0); a.paste(im.split()[3], (30, 38))
        sh.putalpha(a.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.45)))
        sh.alpha_composite(im, (30, 30))
        _ic[k] = pil2f(sh)
    return _ic[k]

_img = {}
def photo(name):
    if name not in _img:
        _img[name] = cv2.cvtColor(cv2.imread(f'{BR}/{name}.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)
    return _img[name]

def cover(im, cw, ch, z=1.0, fx=.5, fy=.5):
    ih, iw = im.shape[:2]
    a = cw / ch
    bw, bh = (ih * a, ih) if iw / ih > a else (iw, iw / a)
    bw, bh = bw / z, bh / z
    cx = min(max(fx * iw, bw / 2), iw - bw / 2); cy = min(max(fy * ih, bh / 2), ih - bh / 2)
    sx, sy = cw / bw, ch / bh
    M = np.float32([[sx, 0, -(cx - bw / 2) * sx], [0, sy, -(cy - bh / 2) * sy]])
    return cv2.warpAffine(im, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

_rm = {}
def rmask(w, h, r):
    k = (w, h, r)
    if k not in _rm:
        m = Image.new('L', (w * 2, h * 2), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=255)
        m = m.resize((w, h), Image.LANCZOS)
        b = Image.new('L', (w * 2, h * 2), 0)
        ImageDraw.Draw(b).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, outline=255, width=6)
        b = b.resize((w, h), Image.LANCZOS)
        sh = Image.new('L', (w + 160, h + 160), 0)
        ImageDraw.Draw(sh).rounded_rectangle((80, 92, w + 79, h + 91), r, fill=255)
        sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(30))).astype(np.float32)
        gl = Image.new('L', (w + 160, h + 160), 0)
        ImageDraw.Draw(gl).rounded_rectangle((80, 80, w + 79, h + 79), r, fill=255)
        gl = np.asarray(gl.filter(ImageFilter.GaussianBlur(40))).astype(np.float32)
        shadow = np.zeros((h + 160, w + 160, 4), np.float32)
        shadow[..., :3] = np.array([60, 130, 255], np.float32) * (gl / 255)[..., None]
        shadow[..., 3] = np.clip(sh * 0.7 + gl * 0.35, 0, 255)
        _rm[k] = (np.asarray(m).astype(np.float32)[..., None] / 255, np.asarray(b).astype(np.float32)[..., None] / 255, shadow)
    return _rm[k]

def card(fr, name, cx, cy, cw, ch, a=1.0, s=1.0, rot=0.0, z=1.0, fx=.5, fy=.5, r=30, crop=None):
    im = photo(name)
    if crop: im = im[crop[1]:crop[3], crop[0]:crop[2]]
    pic = cover(im, cw, ch, z, fx, fy)
    m, b, shadow = rmask(cw, ch, r)
    L = np.zeros((ch, cw, 4), np.float32)
    L[..., :3] = pic * (1 - b * 0.55) + 255 * b * 0.55
    L[..., 3:4] = m * 255
    put(fr, shadow, cx, cy, a, s, rot)
    put(fr, L, cx, cy, a, s, rot)

_tl = {}
def tile(src, label=None, size=210, lab_col=WHITE):
    """glass rounded-square icon tile (src = twemoji codepoint or 'photo:name')"""
    k = (src, label, size, lab_col)
    if k in _tl: return _tl[k]
    S = size
    im = Image.new('RGBA', (S + 80, S + 80 + (64 if label else 0)), (0, 0, 0, 0))
    oy = 64 if label else 0
    sh = Image.new('L', im.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((40, 52 + oy, 40 + S, 52 + oy + S), S // 5, fill=170)
    shl = Image.new('RGBA', im.size, (0, 0, 0, 0)); shl.putalpha(sh.filter(ImageFilter.GaussianBlur(16)))
    im.alpha_composite(shl)
    body = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    g = np.linspace(0, 1, S)[:, None]
    arr = np.zeros((S, S, 4), np.uint8)
    arr[..., 0] = (92 * (1 - g) + 44 * g).astype(np.uint8)
    arr[..., 1] = (128 * (1 - g) + 72 * g).astype(np.uint8)
    arr[..., 2] = (190 * (1 - g) + 130 * g).astype(np.uint8)
    arr[..., 3] = 235
    body = Image.fromarray(arr, 'RGBA')
    mk = Image.new('L', (S, S), 0); ImageDraw.Draw(mk).rounded_rectangle((0, 0, S - 1, S - 1), S // 5, fill=255)
    if src.startswith('photo:'):
        p = cover(photo(src[6:]), S - 28, S - 28).clip(0, 255).astype(np.uint8)
        pm = Image.new('L', (S - 28, S - 28), 0); ImageDraw.Draw(pm).rounded_rectangle((0, 0, S - 29, S - 29), S // 7, fill=255)
        pi = Image.fromarray(p, 'RGB').convert('RGBA'); pi.putalpha(pm)
        body.alpha_composite(pi, (14, 14))
    else:
        isz = int(S * 0.6)
        png = cairosvg.svg2png(url=f'{TW}/{src}.svg', output_width=isz, output_height=isz)
        body.alpha_composite(Image.open(io.BytesIO(png)).convert('RGBA'), ((S - isz) // 2, (S - isz) // 2))
    hl = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle((1, 1, S - 2, S - 2), S // 5, outline=(255, 255, 255, 120), width=3)
    body.alpha_composite(hl)
    body.putalpha(Image.fromarray(np.minimum(np.asarray(body.split()[3]), np.asarray(mk))))
    im.alpha_composite(body, (40, 40 + oy))
    L = pil2f(im)
    if label:
        tl = T(label, FP6, 34, lab_col, shadow=0.5)
        put(L, tl, L.shape[1] / 2, 36)
    _tl[k] = L
    return L


def cursor_layer():
    im = Image.new('RGBA', (120, 140), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pts = [(10, 6), (10, 96), (34, 74), (52, 116), (68, 108), (50, 68), (82, 68)]
    sh = Image.new('L', im.size, 0); ImageDraw.Draw(sh).polygon([(x + 4, y + 6) for x, y in pts], fill=150)
    s = Image.new('RGBA', im.size, (0, 0, 0, 0)); s.putalpha(sh.filter(ImageFilter.GaussianBlur(5)))
    im.alpha_composite(s)
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(15, 15, 25, 255), width=5)
    return pil2f(im)
CURSOR = None

def cursor(fr, t, path, clicks=()):
    """path: [(t, x, y), ...] eased between keys; tip of arrow at (x, y)"""
    global CURSOR
    if CURSOR is None: CURSOR = cursor_layer()
    if t < path[0][0]: return
    x, y = path[-1][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t0 <= t < t1:
            p = ease((t - t0) / (t1 - t0)); x, y = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p; break
    s = 1.0
    for tc in clicks:
        if tc <= t: sfx('click', tc)
        if 0 <= t - tc < 0.35:
            q = (t - tc) / 0.35
            s = 1 - 0.18 * math.sin(math.pi * min(1, q * 2))
            rr = int(20 + 70 * q)
            ov = fr[int(y) - 100:int(y) + 100, int(x) - 100:int(x) + 100]
            if ov.shape[:2] == (200, 200):
                ring = np.zeros((200, 200), np.uint8); cv2.circle(ring, (100, 100), rr, 255, 5, cv2.LINE_AA)
                ov += (255 - ov) * (ring[..., None] / 255.0) * (1 - q) * 0.9
    a = prog(t, path[0][0], 0.15)
    put(fr, CURSOR, x + 33 * s, y + 52 * s, a, s)


# ---------------------------------------------------------------- B-roll pages
def make_bg(seed):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rng = np.random.default_rng(seed)
    cx, cy = W * (0.45 + 0.1 * rng.random()), H * (0.38 + 0.08 * rng.random())
    r = np.sqrt(((xx - cx) / 900) ** 2 + ((yy - cy) / 1250) ** 2)
    g = np.clip(r, 0, 1)[..., None] ** 1.3
    bg = NAVY_C * (1 - g) + NAVY_E * g
    # a soft diagonal light streak like the reference
    st = np.exp(-(((xx * 0.5 + yy * 0.86) - (700 + 400 * rng.random())) / 160) ** 2)
    bg += st[..., None] * np.array([10, 26, 60], np.float32)
    bg += rng.normal(0, 2.0, (H, W, 1)).astype(np.float32)
    # decorations: dashed arcs, sparkle stars, gold swoosh (static, revealed by a wipe)
    deco = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(deco)
    def bez(p0, p1, p2, n=80):
        return [((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
                 (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]) for u in np.linspace(0, 1, n)]
    side = seed % 2
    a = bez((W * side + (-40 if side == 0 else 40), 420 + 200 * rng.random()), (W * 0.5, 120), (W * (1 - side) * 0.9 + 60, -40), 140)
    for i in range(0, len(a) - 1, 4):
        d.line([a[i], a[i + 1]], fill=(255, 255, 255, 150), width=4)
    b = bez((W * (1 - side), H - 640), (W * 0.62 if side else W * 0.38, H - 380), (W * side * 0.7 + 150, H + 40), 120)
    for i in range(0, len(b) - 1, 4):
        d.line([b[i], b[i + 1]], fill=(255, 255, 255, 120), width=4)
    deco = np.asarray(deco).astype(np.float32)
    # gold swoosh in one bottom corner
    sw = np.zeros((H, W), np.float32)
    pts = np.array(bez((-80 if side else W + 80, H - 330), (W * 0.5, H - 70), (W + 80 if side else -80, H - 230), 120), np.int32)
    cv2.polylines(sw, [pts], False, 1.0, 26, cv2.LINE_AA)
    sw = cv2.GaussianBlur(sw, (0, 0), 3) * 0.95 + cv2.GaussianBlur(sw, (0, 0), 24) * 0.8
    gold = np.zeros((H, W, 4), np.float32)
    gx = (xx / W)[..., None]
    gold[..., :3] = np.array([255, 226, 150], np.float32) * (1 - gx) + np.array([196, 132, 40], np.float32) * gx
    gold[..., 3] = np.clip(sw, 0, 1) * 255
    return bg, deco, gold, side
_bgs = {}
def page(seed, lt):
    if seed not in _bgs: _bgs[seed] = make_bg(seed)
    bg, deco, gold, side = _bgs[seed]
    fr = bg.copy()
    wipe = ease(lt / 0.6)
    for L in (deco, gold):
        a = L[..., 3:4] / 255
        if wipe < 1:
            ramp = np.clip((np.linspace(0, 1, W) if side == 0 else np.linspace(1, 0, W)) < wipe, 0, 1).astype(np.float32)
            a = a * ramp[None, :, None]
        fr += (L[..., :3] - fr) * a
    # two twinkling 4-point stars
    rng = np.random.default_rng(seed + 11)
    for k in range(2):
        sx, sy = (120 + 840 * rng.random(), 160 + 260 * rng.random()) if k == 0 else ((60 + 110 * rng.random()) if rng.random() < .5 else (910 + 110 * rng.random()), 1090 + 60 * rng.random())
        star(fr, sx, sy, 34 + 14 * k, lt * 40 + k * 30, 0.6 + 0.4 * math.sin(lt * 4 + k))
    return fr

_star = {}
def star(fr, x, y, r, rot, a):
    if r not in _star:
        S = int(r * 3); im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        c = S / 2; w_ = r * 0.22
        d.polygon([(c, c - r), (c + w_, c - w_), (c + r, c), (c + w_, c + w_), (c, c + r), (c - w_, c + w_), (c - r, c), (c - w_, c - w_)], fill=(255, 255, 255, 255))
        g = im.filter(ImageFilter.GaussianBlur(r * 0.25))
        out = Image.alpha_composite(g, im)
        _star[r] = pil2f(out)
    put(fr, _star[r], x, y, a, 1.0, rot)


def dashed(fr, p0, p1, t, t0, dur=0.35, col=(255, 255, 255), w=6, head=True):
    """marching-ants arrow that draws on from p0 to p1"""
    p = ease((t - t0) / dur)
    if p <= 0: return
    x0, y0 = p0; x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
    ov = np.zeros((H, W), np.float32)
    off = (t * 90) % 36
    s = -off
    while s < L * p:
        a_, b_ = max(0, s), min(L * p, s + 20)
        if b_ > a_:
            cv2.line(ov, (int(x0 + ux * a_), int(y0 + uy * a_)), (int(x0 + ux * b_), int(y0 + uy * b_)), 1.0, w, cv2.LINE_AA)
        s += 36
    if head and p > 0.95:
        ex, ey = x0 + ux * L, y0 + uy * L
        nx, ny = -uy, ux
        tri = np.array([(ex + ux * 6, ey + uy * 6), (ex - ux * 26 + nx * 18, ey - uy * 26 + ny * 18), (ex - ux * 26 - nx * 18, ey - uy * 26 - ny * 18)], np.int32)
        cv2.fillPoly(ov, [tri], 1.0, cv2.LINE_AA)
    fr += (np.array(col, np.float32) - fr) * ov[..., None] * 0.95


def knob(fr, cx, cy, v, label='GAIN', R=170):
    """amplifier gain knob, v in 0..1"""
    S = R * 2 + 140
    im = Image.new('RGBA', (S * 2, S * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = S; r2 = R * 2
    d.arc((c - r2 - 40, c - r2 - 40, c + r2 + 40, c + r2 + 40), 135, 405, fill=(255, 255, 255, 50), width=26)
    steps = max(1, int(270 * v / 4))
    for i in range(steps):
        u0 = i / steps * v; u1 = (i + 1) / steps * v
        col = (np.array([80, 220, 120]) * (1 - min(1, u0 * 2)) + np.array([255, 205, 60]) * min(1, u0 * 2)) if u0 < .5 else \
              (np.array([255, 205, 60]) * (1 - (u0 - .5) * 2) + np.array([255, 60, 50]) * (u0 - .5) * 2)
        d.arc((c - r2 - 40, c - r2 - 40, c + r2 + 40, c + r2 + 40), 135 + 270 * u0, 135 + 270 * u1 + 0.6, fill=tuple(int(x) for x in col) + (255,), width=26)
    for i in range(11):
        an = math.radians(135 + 27 * i)
        d.line([(c + math.cos(an) * (r2 + 6), c + math.sin(an) * (r2 + 6)), (c + math.cos(an) * (r2 - 18), c + math.sin(an) * (r2 - 18))], fill=(255, 255, 255, 200), width=6)
    d.ellipse((c - r2 + 30, c - r2 + 30, c + r2 - 30, c + r2 - 30), fill=(30, 36, 56, 255), outline=(140, 160, 200, 255), width=8)
    d.ellipse((c - r2 + 70, c - r2 + 70, c + r2 - 70, c + r2 - 70), fill=(46, 54, 80, 255))
    an = math.radians(135 + 270 * v)
    d.line([(c + math.cos(an) * 40, c + math.sin(an) * 40), (c + math.cos(an) * (r2 - 90), c + math.sin(an) * (r2 - 90))], fill=(255, 150, 40, 255), width=20)
    im = im.resize((S, S), Image.LANCZOS)
    L = pil2f(im)
    put(fr, L, cx, cy)
    put(fr, T('MIN', FP7, 30, shadow=0.4), cx - R * 0.78, cy + R * 0.98)
    put(fr, T('MAX', FP7, 30, (255, 110, 90), shadow=0.4), cx + R * 0.78, cy + R * 0.98)
    put(fr, T(label, FM8, 44, track=6), cx, cy + R + 52)


def meter(fr, cx, cy, v, n=12, bw=70, bh=26, gap=8):
    """vertical LED level meter, v in 0..1"""
    for i in range(n):
        u = i / (n - 1)
        col = (80, 220, 120) if u < .55 else (255, 205, 60) if u < .8 else (255, 60, 50)
        on = u <= v + 1e-3
        y = int(cy + (n / 2 - i) * (bh + gap)); x = int(cx - bw / 2)
        reg = fr[y:y + bh, x:x + bw]
        reg += (np.array(col if on else (60, 80, 120), np.float32) - reg) * (0.95 if on else 0.5)


def pill(fr, text, cx, cy, a=1.0, fill=(255, 140, 30), tcol=WHITE, size=40, outline=False):
    L = T(text, FM8, size, tcol, shadow=0.0, track=4)
    tw = ink_w(L, size); ph = int(size * 1.9); pw = tw + int(size * 1.6)
    im = Image.new('RGBA', (pw + 40, ph + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if outline:
        d.rounded_rectangle((20, 20, pw + 19, ph + 19), ph // 2, fill=(255, 255, 255, 40), outline=(255, 255, 255, 230), width=4)
    else:
        d.rounded_rectangle((20, 20, pw + 19, ph + 19), ph // 2, fill=fill + (245,))
    put(fr, pil2f(im), cx, cy, a)
    put(fr, L, cx, cy - size * 0.06, a)


# ---------------------------------------------------------------- person cutout
_interp = None
def seg(img):
    global _interp
    if _interp is None:
        _interp = Interpreter(model_path=SEG_MODEL); _interp.allocate_tensors()
    i = _interp.get_input_details()[0]; o = _interp.get_output_details()[0]
    x = cv2.resize(img, (256, 256)).astype(np.float32)[None] / 255
    _interp.set_tensor(i['index'], x); _interp.invoke()
    m = _interp.get_tensor(o['index'])[0, ..., 0]
    m = np.clip((m - 0.35) / 0.3, 0, 1)
    m = cv2.GaussianBlur(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 2.5)
    return m[..., None]


# ---------------------------------------------------------------- edit plan
# A shots: ('A', s, e, lead, key, tail, extras)  /  B shots: ('B', s, e, page_fn)
def A(s, e, lead, key, tail='', **kw): return ('A', s, e, lead, key, tail, kw)

def b_fluct(fr, t, s, e):
    lt = t - s
    put(fr, emoji('1f50a', 260), W / 2 - 150, 760, prog(t, s, .2), 0.6 + 0.4 * back(prog(t, s + .05, .3)))
    for k in range(3):
        on = math.sin(lt * 9 + k) > -0.2 and int(lt * 5) % 3 != 2
        v = (0.35 + 0.55 * abs(math.sin(lt * 7 + k * 1.3))) if on else 0.05
        meter(fr, W / 2 + 90 + k * 95, 700, v, n=10, bw=64, bh=24)
    caption(fr, t, s + 0.1, 'sat-sat ada, sat-sat hilang', 'TENGGELAM', '', 1250)

def b_sub(fr, t, s, e):
    lt = t - s
    sw = 8.6
    if t < sw:
        p = ease(lt / .35)
        card(fr, 'web2', W / 2, 720 + (1 - p) * 60, 900, 680, p, 0.96 + 0.04 * p, -2, 1.0 + 0.04 * lt / 2)
    else:
        q = ease((t - sw) / .3)
        card(fr, 'web2', W / 2 - 700 * q, 720, 900, 680, 1 - q, 1, -2)
        card(fr, 'web6', W / 2 + 700 * (1 - q), 720, 900, 680, q, 1, 2, 1.0 + 0.05 * (t - sw) / 2)
        if t >= sw: sfx('whoosh', sw)
    put(fr, T('Subwoofer', FP7, 50), 250, 300, prog(t, s + .1, .2))
    caption(fr, t, s + 0.15, 'subwoofer keluarkan', 'FREKUENSI RENDAH', '', 1240, ksize=112)
    cursor(fr, t, [(s + .5, 980, 1300), (s + 1.0, 720, 900), (sw + .4, 760, 860)], clicks=(s + 1.05,))

def b_bass(fr, t, s, e):
    lt = t - s
    card(fr, '1', W / 2, 760, 960, 1000, 1, 1.0 + 0.06 * lt, 0, 1.0, crop=(0, 60, 630, 853))
    caption(fr, t, s, '', 'BASS', '', 1330, ksize=220)

def b_vocal(fr, t, s, e):
    put(fr, tile('photo:web4', 'Speaker Pintu', 300), 290, 760, prog(t, s, .2), 0.6 + 0.4 * back(prog(t, s, .3)))
    dashed(fr, (480, 790), (610, 790), t, s + .45)
    put(fr, tile('1f3a4', 'Vokal', 300, (255, 196, 90)), 800, 760, prog(t, s + .8, .2), 0.6 + 0.4 * back(prog(t, s + .8, .3)))
    if t >= s + .8: sfx('pop', s + .8)
    caption(fr, t, s + .1, 'speaker pintu keluarkan', 'VOKAL', 'bunyi suara dalam lagu', 1230, tail_at=1.3)

def b_graph(fr, t, s, e):
    lt = t - s
    cw, ch = 980, 668
    p = ease(lt / .35)
    card(fr, '3', W / 2, 760 + (1 - p) * 50, cw, ch, p, 1, 0, 1.0 + 0.03 * lt / 2, r=24)
    # selection box sweeps to the high-frequency end of the graph
    q = ease((t - s - .6) / .7)
    if q > 0:
        x0 = W / 2 - cw / 2 + cw * (0.10 + 0.52 * q); x1 = x0 + cw * 0.27
        y0, y1 = 760 - ch / 2 + 70, 760 + ch / 2 - 90
        ov = np.zeros((H, W), np.float32)
        cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 6, cv2.LINE_AA)
        fr += (np.array(ORANGE_BOT, np.float32) - fr) * ov[..., None]
        reg = fr[int(y0):int(y1), int(x0):int(x1)]
        reg += (np.array([255, 160, 60], np.float32) - reg) * 0.12
        cursor(fr, t, [(s + .5, x0 + 60, y1 - 40), (s + 1.3, x1 - 30, y1 - 40)], clicks=(s + .55,))
    caption(fr, t, s + .1, 'frekuensi yang', 'TINGGI', '', 1240)

def b_over(fr, t, s, e):
    lt = t - s
    p = ease(lt / .3)
    sh = 6 * math.sin(lt * 60) if lt > 0.9 else 0
    card(fr, 'web6', 400 + sh, 740, 640, 760, p, 1, -3, 1.0 + 0.05 * lt)
    v = clip01(lt / 0.9)
    meter(fr, 870 + sh, 700, v, n=14, bw=110, bh=30)
    if lt > 0.9:
        sfx('boom', s + .9)
        put(fr, T('MAX!', FM9I, 80, (255, 70, 60), glow=0.4), 870, 340, 0.6 + 0.4 * math.sin(lt * 18))
    caption(fr, t, s + .1, 'subwoofer dibuka', 'OVER SANGAT', '', 1250)

def b_scale(fr, t, s, e, tilt_to=14, ok=False):
    """balance beam; positive tilt_to = left side heavy"""
    lt = t - s
    ang = tilt_to * back(clip01((lt - .3) / .6)) if not ok else tilt_to * (1 - ease((lt - .4) / .8))
    cx, cy = W / 2, 560
    im = Image.new('RGBA', (W, 600), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(cx - 70, 560), (cx + 70, 560), (cx, 40)], fill=(70, 110, 175, 255))
    d.rounded_rectangle((cx - 120, 548, cx + 120, 580), 12, fill=(90, 130, 200, 255))
    put(fr, pil2f(im), W / 2, cy + 230, prog(t, s, .2))
    a = math.radians(ang); half = 400; py = cy + 40
    lx, ly = cx - math.cos(a) * half, py + math.sin(a) * half
    rx, ry = cx + math.cos(a) * half, py - math.sin(a) * half
    ov = np.zeros((H, W), np.float32)
    cv2.line(ov, (int(lx), int(ly)), (int(rx), int(ry)), 1.0, 18, cv2.LINE_AA)
    cv2.circle(ov, (int(cx), int(cy + 40)), 22, 1.0, -1, cv2.LINE_AA)
    for x_, y_ in ((lx, ly), (rx, ry)):
        cv2.line(ov, (int(x_), int(y_)), (int(x_), int(y_ + 90)), 0.8, 4, cv2.LINE_AA)
    fr += (np.array([235, 240, 255], np.float32) - fr) * ov[..., None] * prog(t, s, .2)
    if ok:
        put(fr, tile('1f50a', 'Bass', 240), lx, ly + 190, prog(t, s + .1, .2))
        put(fr, tile('1f3a4', 'Vokal', 240, (255, 196, 90)), rx, ry + 190, prog(t, s + .25, .2))
        if lt > 1.8:
            sfx('ding', s + 1.8)
            put(fr, emoji('2705', 150), cx, cy - 110, 1, 0.6 + 0.4 * back(prog(t, s + 1.8, .3)))
    else:
        put(fr, tile('photo:web2', 'Subwoofer', 240), lx, ly + 190, prog(t, s + .1, .2))
        put(fr, tile('photo:web4', 'Speaker Pintu', 240, (255, 196, 90)), rx, ry + 190, prog(t, s + .25, .2))
        put(fr, emoji('2696', 130), cx, cy - 120, prog(t, s, .2), 0.6 + 0.4 * back(prog(t, s, .3)))

def b_balancing(fr, t, s, e):
    b_scale(fr, t, s, e, 14)
    caption(fr, t, s + .1, 'sebenarnya masalah', 'SETUP BALANCING', 'audio kereta dah "lari"', 1290, tail_at=2.2, ksize=118)

def b_gain(fr, t, s, e):
    lt = t - s
    p = ease(lt / .3)
    card(fr, 'web3', W / 2 - 120, 560 + (1 - p) * 40, 760, 560, p, 1, -3, 1.0 + 0.05 * lt / 1.8)
    q = prog(t, s + .35, .35)
    if q > 0:
        sfx('pop', s + .35)
        knob(fr, 760, 900, 0.35 + 0.08 * math.sin(lt * 3), R=int(140 * (0.4 + 0.6 * back(q))))
    caption(fr, t, s + .1, 'terutama bahagian', 'GAIN', '', 1300, ksize=170)

def b_signal(fr, t, s, e):
    lt = t - s
    xs = [190, 540, 890]; y = 760
    for k, (src, lab) in enumerate((('photo:web7', 'Head Unit'), ('photo:web3', 'Amplifier'), ('photo:web2', 'Subwoofer'))):
        tk = s + k * .45
        put(fr, tile(src, lab, 240, (255, 196, 90) if k == 1 else WHITE), xs[k], y, prog(t, tk, .2), 0.6 + 0.4 * back(prog(t, tk, .3)))
        if t >= tk: sfx('pop', tk)
    dashed(fr, (330, y + 32), (400, y + 32), t, s + .35, .25)
    dashed(fr, (680, y + 32), (750, y + 32), t, s + .8, .25)
    # glowing signal dot travelling through the chain
    if lt > 1.2:
        u = ((lt - 1.2) / 1.2) % 1
        x = xs[0] + (xs[2] - xs[0]) * u
        ov = np.zeros((H, W), np.float32); cv2.circle(ov, (int(x), int(y + 32)), 16, 1.0, -1, cv2.LINE_AA)
        ov = np.maximum(ov, cv2.GaussianBlur(ov, (0, 0), 14) * 2.2).clip(0, 1)
        fr += (np.array([255, 190, 80], np.float32) - fr) * ov[..., None]
    put(fr, T('Signal', FP7, 50), 200, 470, prog(t, s + .1, .2))
    caption(fr, t, s + .2, 'kawal signal', 'KELUAR MASUK', 'ke subwoofer', 1220)
    cursor(fr, t, [(s + 1.0, 1000, 1200), (s + 1.6, 560, 820)], clicks=(s + 1.65,))

def b_max(fr, t, s, e):
    lt = t - s
    cw, ch = 980, 653
    p = ease(lt / .3)
    card(fr, '4', W / 2, 740 + (1 - p) * 40, cw, ch, p, 1, 0, 1.0 + 0.04 * lt / 1.5, r=24)
    # highlight the 'can cause damage' corner of the infographic
    q = ease((t - s - .5) / .3)
    if q > 0:
        x0, y0 = W / 2 + cw * 0.22, 740 + ch * 0.36; x1, y1 = W / 2 + cw * 0.5 - 6, 740 + ch * 0.5 - 6
        ov = np.zeros((H, W), np.float32)
        cv2.rectangle(ov, (int(x0), int(y0)), (int(x0 + (x1 - x0) * q), int(y1)), 1.0, 6, cv2.LINE_AA)
        fr += (np.array([255, 70, 60], np.float32) - fr) * ov[..., None]
        put(fr, emoji('26a0', 110), x1 - 20, y0 - 50, q, 0.6 + 0.4 * back(q))
    caption(fr, t, s + .05, 'sampai', 'MAKSIMUM', '', 1290, ksize=150)

def b_knobmax(fr, t, s, e):
    lt = t - s
    v = 0.3 + 0.7 * ease(lt / 0.8)
    sh = 5 * math.sin(lt * 55) if lt > 0.8 else 0
    knob(fr, W / 2 + sh, 680, v, R=200)
    if lt > 0.9:
        sfx('pop', s + .9)
        put(fr, emoji('274c', 200), W / 2 + 230, 470, 1, 0.5 + 0.5 * back(prog(t, s + .9, .3)), -8)
    caption(fr, t, s + .05, 'gain maksimum', 'BUKAN BASS PADU', '', 1260, ksize=118)

def b_img1(fr, t, s, e):
    lt = t - s
    p = ease(lt / .3)
    card(fr, '1', W / 2, 740 + (1 - p) * 40, 1000, 667, p, 1, 0, 1.0 + 0.06 * lt / 2.5, r=26)
    caption(fr, t, s + .1, 'ada yang keluarkan', 'BASS', '', 1260, ksize=170)

def b_balance_ok(fr, t, s, e):
    b_scale(fr, t, s, e, 14, ok=True)
    caption(fr, t, s + .1, 'setup bagi', 'SAMA-SAMA BALANCE', 'ada bass, ada vokal yang sedap', 1290, tail_at=1.8, ksize=110)

def b_img2(fr, t, s, e):
    lt = t - s
    p = ease(lt / .3)
    card(fr, '2', W / 2, 740 + (1 - p) * 40, 1000, 667, p, 1, 0, 1.0 + 0.05 * lt / 3.6, r=26)
    for k, (x, y) in enumerate(((150, 400), (930, 1030), (980, 420))):
        tk = s + 0.5 + k * 0.5
        if t >= tk: sfx('shine', tk)
        put(fr, emoji('2728', 110), x, y + 8 * math.sin(3 * lt + k), prog(t, tk, .2), 0.6 + 0.4 * back(prog(t, tk, .3)))
    caption(fr, t, s + .1, 'bass padu,', 'VOKAL KELUAR', 'baru sedap', 1260, tail_at=2.4)

def b_follow(fr, t, s, e, src=None):
    lt = t - s
    if src is not None:
        R = 230
        face = src[460:1100, 160:800].astype(np.float32)
        face = cv2.resize(face, (2 * R, 2 * R), interpolation=cv2.INTER_AREA)
        L = np.zeros((2 * R + 24, 2 * R + 24, 4), np.float32)
        m = np.zeros((2 * R + 24, 2 * R + 24), np.float32)
        cv2.circle(m, (R + 12, R + 12), R + 10, 255, -1, cv2.LINE_AA)
        L[..., :3] = 255; L[..., 3] = m
        put(fr, L, 380, 700, prog(t, s, .2), 0.6 + 0.4 * back(prog(t, s, .35)))
        m2 = np.zeros((2 * R, 2 * R), np.float32); cv2.circle(m2, (R, R), R - 2, 255, -1, cv2.LINE_AA)
        L2 = np.dstack([face, m2])
        put(fr, L2, 380, 700, prog(t, s, .2), 0.6 + 0.4 * back(prog(t, s, .35)))
    for k, cp in enumerate(('1f44d', '2764', '1f4ac')):
        tk = s + .4 + k * .3
        if t >= tk: sfx('pop', tk)
        put(fr, tile(cp, None, 170), 820, 470 + k * 230, prog(t, tk, .2), 0.6 + 0.4 * back(prog(t, tk, .3)))
    q = prog(t, s + 1.5, .3)
    if q > 0:
        pill(fr, 'FOLLOW', 380, 1040, q, size=48)
    cursor(fr, t, [(s + 1.6, 900, 1250), (s + 2.4, 430, 1060)], clicks=(s + 2.5,))
    caption(fr, t, s + .1, 'jangan lupa', 'FOLLOW & LIKE', 'dan komen sekali', 1250, tail_at=2.0)

PLAN = [
    A(0.00, 1.92, 'pernah tak buka lagu', 'BASS KUAT?', ghost='BASS', emo=('1f3b5', 0.35)),
    A(1.92, 3.80, 'tiba-tiba sound', 'VOKAL', 'hampa...', emo=('1f3a4', 0.3)),
    ('B', 3.80, 5.70, b_fluct),
    A(5.70, 6.75, 'okay,', 'DENGAR SINI', emo=('1f442', 0.2)),
    ('B', 6.75, 10.45, b_sub),
    ('B', 10.45, 11.25, b_bass),
    A(11.25, 13.42, 'pula,', 'SPEAKER PINTU', behind_card=('web4', 0.1)),
    ('B', 13.42, 16.45, b_vocal),
    ('B', 16.45, 18.72, b_graph),
    A(18.72, 21.38, 'problem dia sekarang, bila ada', 'SUBWOOFER', emo=('26a0', 1.4)),
    ('B', 21.38, 23.20, b_over),
    A(23.20, 25.10, 'speaker pintu semua', 'TENGGELAM', emo=('1f507', 0.5)),
    A(25.10, 26.70, 'sebenarnya', 'BUKAN', 'masalah speaker pintu', emo=('274c', 0.3)),
    ('B', 26.70, 30.82, b_balancing),
    A(30.82, 34.02, 'kena faham cara', 'SETUP SUBWOOFER', 'betul-betul', emo=('1f527', 0.6)),
    ('B', 34.02, 35.80, b_gain),
    A(35.80, 36.45, '', 'GAIN TU APA?', emo=('1f914', 0.05)),
    ('B', 36.45, 39.28, b_signal),
    A(39.28, 40.50, 'jangan pusing', 'GAIN', emo=('26a0', 0.3)),
    ('B', 40.50, 42.02, b_max),
    ('B', 42.02, 44.00, b_knobmax),
    A(44.00, 46.20, 'dalam kereta, ada yang keluarkan', 'BUNYI', emo=('1f3b6', 0.5)),
    ('B', 46.20, 48.70, b_img1),
    A(48.70, 51.45, 'bass kuat,', 'BUNYI TENGGELAM', emo=('1f635', 0.6)),
    A(51.45, 52.86, 'oh,', 'TAK BALANCE', 'sayang', emo=('2696', 0.2)),
    ('B', 52.86, 56.90, b_balance_ok),
    A(56.90, 59.60, 'senang cerita, kena', 'SETUP BALANCE', emo=('2705', 1.0), ghost='BALANCE'),
    ('B', 59.60, 63.20, b_img2),
    A(63.20, 64.30, '', 'BARU SEDAP', 'okay?', emo=('1f44c', 0.1)),
    ('B', 64.30, 68.55, b_follow),
    A(68.55, 71.70, 'and', 'KOMEN', 'lah sekali', emo=('1f4ac', 0.3), handle=True),
    A(71.70, 73.10, 'okay lah,', 'BYE!', emo=('1f44b', 0.3), handle=True),
]
HANDLE = '@athirahmajidd_'


def shot_at(t):
    for i, sh in enumerate(PLAN):
        if sh[1] <= t < sh[2]: return i
    return len(PLAN) - 1


# ---------------------------------------------------------------- A-roll
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.55 * np.clip(np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H * 0.45) / (H * 0.62)) ** 2) - 0.45, 0, 1) ** 1.5)[..., None]
BOTG = (1 - 0.45 * np.clip((np.arange(H) - 1050) / 870, 0, 1) ** 1.3)[:, None, None].astype(np.float32)
del yy, xx

def grade(fr):
    fr = (fr - 118) * 1.07 + 118
    lum = fr.mean(axis=2, keepdims=True)
    fr = lum + (fr - lum) * 0.9
    fr *= np.array([0.95, 0.99, 1.07], np.float32)
    fr += (1 - np.clip(lum / 255, 0, 1)) * np.array([-4, 4, 18], np.float32)
    return fr * VIG * BOTG


def a_frame(src, i, t):
    _, s, e, lead, key, tail, kw = PLAN[i]
    lt = t - s
    z = (1.0 if i % 2 == 0 else 1.09) + 0.025 * lt / max(e - s, 1)
    fx, fy = 520, 820
    M = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy]])
    img = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR)
    base = grade(img.astype(np.float32))
    if 'ghost' in kw or 'behind_card' in kw:
        m = seg(img)
        over = base.copy()
        if 'ghost' in kw:
            p = ease(lt / .4)
            L = T(kw['ghost'], FM9I, 290, 'orange', glow=0.4, shadow=0, maxw=1000)
            put(over, L, W / 2, 470 - (1 - p) * 40, 0.8 * p, 1.15 - 0.15 * p)
        if 'behind_card' in kw:
            name, d = kw['behind_card']
            p = ease((lt - d) / .4)
            if p > 0:
                if lt >= d: sfx('whoosh', s + d)
                card(over, name, 860 + (1 - p) * 300, 520, 400, 500, p, 1, 6, 1.0 + 0.04 * lt)
        base = over * (1 - m) + base * m
    fr = base
    if 'emo' in kw:
        cp, d = kw['emo']
        p = prog(t, s + d, .3)
        if p > 0:
            sfx('pop', s + d)
            put(fr, emoji(cp, 150), 960, 1150 + 10 * math.sin(lt * 3.2), min(1, p * 2), 0.5 + 0.5 * back(p), 8 * math.sin(lt * 2))
    caption(fr, t, s + 0.03, lead, key, tail, 1210, tail_at=None)
    if t < 3.8:
        pill(fr, 'TIPS AUDIO KERETA', W / 2, 210, ease(t / .3), size=34, outline=True)
    if kw.get('handle'):
        p = ease((t - 68.85) / .3)
        pill(fr, HANDLE, W / 2, 1680, p, fill=(255, 255, 255), tcol=(20, 30, 60), size=36)
    return fr


# ---------------------------------------------------------------- main loop
def frame_at(src, t):
    i = shot_at(t)
    sh = PLAN[i]
    if sh[0] == 'A':
        fr = a_frame(src, i, t)
    else:
        fr = page(i, t - sh[1])
        if sh[3] is b_follow: b_follow(fr, t, sh[1], sh[2], src)
        else: sh[3](fr, t, sh[1], sh[2])
    # cut transition: 4-frame zoom punch with motion smear
    lt = t - sh[1]
    if i > 0:
        sfx('whoosh', sh[1])
        if lt < 0.14:
            p = lt / 0.14
            z = 1 + 0.09 * (1 - p) ** 2
            M = np.float32([[z, 0, W / 2 - z * W / 2], [0, z, H / 2 - z * H / 2]])
            fr = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR)
            bl = cv2.blur(fr, (1, 61))
            fr = fr * p + bl * (1 - p)
    return np.clip(fr, 0, 255).astype(np.uint8)


def decoder(start=0.0):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', SRC, '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)

if os.environ.get('TEST'):
    ts = [float(x) for x in os.environ['TEST'].split(',')]
    tw = int(os.environ.get('TW', 300))
    tiles = []
    for t in ts:
        p = decoder(t)
        src = np.frombuffer(p.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3)
        p.kill()
        im = cv2.resize(frame_at(src, t), (tw, tw * 16 // 9), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f'{t:.2f}', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(im)
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(OUT, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    sys.exit()

dec = decoder()
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                        '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', OUT],
                       stdin=subprocess.PIPE)
n = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    src = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    enc.stdin.write(frame_at(src, n / FPS).tobytes())
    n += 1
    if n % 150 == 0: print(n, flush=True)
enc.stdin.close(); enc.wait()
print('frames', n)
if SFX_OUT:
    json.dump(sorted(SFX, key=lambda x: x[1]), open(SFX_OUT, 'w'))
