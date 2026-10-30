"""PROMO 11.11 — bright festive commercial re-edit of the car-audio talking-head promo (IMG_5148.MOV).

A-roll = brightened original footage with punch-in jump cuts and chest captions.
B-roll = product cards cut from the footage, Mixkit stock clips (free licence) and designed
11.11 / offer / end cards in a purple-pink-yellow sale palette, with confetti bursts.

Usage: python3 render.py SRC.MOV OUT_VIDEO_ONLY.mp4      (env TEST=t1,t2 -> work/test.jpg, source times)
Needs work/fonts/{anton,archivo-black}.ttf (@fontsource) and work/mixkit/hd_{72,41174,4064}.mp4.
"""
import os, subprocess, sys, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
FPS_N, FPS_D = 60000, 1001
FPS = FPS_N / FPS_D
T0, T1 = 0.60, 28.90            # source range kept (trims dead air head/tail)

PURPLE = (96, 66, 224)
PURPLE_D = (62, 36, 170)
INKP = (36, 18, 92)
PINK = (255, 79, 163)
YELLOW = (255, 225, 77)
CYAN = (63, 224, 232)
RED = (230, 36, 44)
WHITE = (255, 255, 255)
CONF = [YELLOW, PINK, CYAN, WHITE, (140, 110, 255), (255, 150, 60)]

ANTON = HERE + '/work/fonts/anton.ttf'
ARCH = HERE + '/work/fonts/archivo-black.ttf'
ID = '/usr/share/fonts/opentype/inter/'
MIX = HERE + '/work/mixkit/'
GRADE = 'null'                  # keep the footage's original colour (user request)

_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc: _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_back(x):
    x = min(max(x, 0), 1); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

# ---------------------------------------------------------------- text helpers
_tc = {}
def text(txt, path, size, fill=WHITE, track=0, shadow=0.55, pad=40):
    k = (txt, path, size, fill, track, shadow)
    if k in _tc: return _tc[k]
    f = font(path, size)
    widths = [f.getlength(c) for c in txt]
    tw = int(sum(widths) + track * (len(txt) - 1))
    asc, desc = f.getmetrics()
    im = Image.new('RGBA', (tw + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = pad
    for c, w in zip(txt, widths):
        d.text((x, pad), c, font=f, fill=fill + (255,)); x += w + track
    bb = im.getbbox()
    im = im.crop((max(bb[0] - pad, 0), max(bb[1] - pad, 0), min(bb[2] + pad, im.width), min(bb[3] + pad, im.height)))
    if shadow: im = with_shadow(im, shadow)
    _tc[k] = im
    return im

def with_shadow(im, op, blur=10, dy=6):
    a = im.getchannel('A').filter(ImageFilter.GaussianBlur(blur))
    sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
    sh.putalpha(a.point(lambda v: int(v * op)))
    base = Image.new('RGBA', im.size, (0, 0, 0, 0))
    base.alpha_composite(sh, (0, dy))
    base.alpha_composite(im)
    return base

def text3d(txt, path, size, front, side, outline=INKP, depth=14, stroke=9):
    """Chunky sticker-style 3D type: dark outline, extruded side, bright face."""
    k = ('3d', txt, path, size, front, side, depth)
    if k in _tc: return _tc[k]
    f = font(path, size)
    bb = f.getbbox(txt, stroke_width=stroke)
    pad = 40
    im = Image.new('RGBA', (bb[2] - bb[0] + depth + 2 * pad, bb[3] - bb[1] + depth + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y = pad - bb[0], pad - bb[1]
    for j in range(depth, -1, -1):
        d.text((x + j, y + j), txt, font=f, fill=outline + (255,), stroke_width=stroke, stroke_fill=outline + (255,))
    for j in range(depth, 0, -1):
        d.text((x + j, y + j), txt, font=f, fill=side + (255,))
    d.text((x, y), txt, font=f, fill=front + (255,), stroke_width=stroke, stroke_fill=outline + (255,))
    d.text((x, y), txt, font=f, fill=front + (255,))
    im = with_shadow(im, 0.35, 14, 10)
    _tc[k] = im
    return im

def pill(txt, path, size, fg=INKP, bg=YELLOW, padx=34, pady=16, outline=INKP):
    k = ('pill', txt, path, size, fg, bg)
    if k in _tc: return _tc[k]
    t = text(txt, path, size, fg, shadow=0, pad=4)
    w, h = t.width + 2 * padx, t.height + 2 * pady
    im = Image.new('RGBA', (w + 60, h + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((36, 38, 30 + w + 6, 30 + h + 8), h // 2, fill=outline + (255,))
    d.rounded_rectangle((30, 30, 30 + w, 30 + h), h // 2, fill=bg + (255,), outline=outline + (255,), width=5)
    im.alpha_composite(t, (30 + padx, 30 + pady))
    _tc[k] = im
    return im

def paste(canvas, im, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    """Composite RGBA `im` centred at (cx, cy), crop-safe."""
    if alpha <= 0.003 or scale <= 0.01: return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    if rot: im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 0.999:
        im = im.copy(); im.putalpha(im.getchannel('A').point(lambda v: int(v * alpha)))
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    sx, sy = max(0, -x), max(0, -y)
    ex, ey = min(im.width, W - x), min(im.height, H - y)
    if ex <= sx or ey <= sy: return
    canvas.alpha_composite(im.crop((sx, sy, ex, ey)), (x + sx, y + sy))

# ---------------------------------------------------------------- decorations
def star_pts(cx, cy, r, inner=0.32):
    pts = []
    for j in range(8):
        a = -math.pi / 2 + j * math.pi / 4
        rr = r if j % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts

def squiggle(d, x0, y0, length, amp, col, waves=2.5, vertical=False, width=12):
    pts = []
    for j in range(60):
        u = j / 59
        a = x0 + u * length if not vertical else x0 + amp * math.sin(u * waves * 2 * math.pi)
        b = y0 + amp * math.sin(u * waves * 2 * math.pi) if not vertical else y0 + u * length
        pts.append((a, b))
    d.line(pts, fill=INKP + (255,), width=width + 8, joint='curve')
    d.line(pts, fill=col + (255,), width=width, joint='curve')

def sale_badge():
    """Original 11.11 SALE sticker in the user's reference palette (drawn from scratch, no stock art)."""
    im = Image.new('RGBA', (1000, 780), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rib = [(95, 330), (905, 255), (930, 560), (70, 610)]
    d.polygon([(x + 10, y + 12) for x, y in rib], fill=INKP + (255,))
    d.polygon(rib, fill=PINK + (255,), outline=INKP + (255,), width=10)
    squiggle(d, 40, 150, 170, 22, CYAN, 2.0)
    squiggle(d, 800, 640, 170, 20, YELLOW, 2.0)
    for (cx, cy, r, c) in [(860, 110, 40, CYAN), (130, 560, 34, YELLOW), (900, 420, 24, WHITE), (70, 330, 20, WHITE)]:
        d.polygon([(x + 4, y + 5) for x, y in star_pts(cx, cy, r)], fill=INKP + (255,))
        d.polygon(star_pts(cx, cy, r), fill=c + (255,), outline=INKP + (255,))
    a = text3d('11.11', ARCH, 250, YELLOW, (150, 96, 255), depth=16, stroke=10)
    b = text3d('SALE', ARCH, 230, WHITE, (150, 96, 255), depth=16, stroke=10)
    im.alpha_composite(a, (500 - a.width // 2, 10))
    im.alpha_composite(b, (500 - b.width // 2, 330))
    return im
BADGE = sale_badge()

def gift_stamp(size=640):
    """Distressed rubber stamp reading FREE GIFT (drawn from scratch)."""
    S = 1000
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = RED + (255,)
    d.ellipse((60, 60, S - 60, S - 60), outline=c, width=46)
    d.ellipse((150, 150, S - 150, S - 150), outline=c, width=14)
    for j in range(16):
        a = j / 16 * 2 * math.pi
        if abs(math.sin(a)) < 0.42: continue  # leave room for the banner
        d.polygon(star_pts(S / 2 + 360 * math.cos(a), S / 2 + 360 * math.sin(a), 30, 0.42), fill=c)
    ban = Image.new('RGBA', (S, 300), (0, 0, 0, 0))
    bd = ImageDraw.Draw(ban)
    bd.rounded_rectangle((10, 20, S - 10, 280), 30, fill=(255, 255, 255, 255), outline=c, width=26)
    bd.rounded_rectangle((44, 52, S - 44, 248), 18, outline=c, width=8)
    t = text('FREE GIFT', ANTON, 190, RED, track=6, shadow=0, pad=4)
    ban.alpha_composite(t, ((S - t.width) // 2, (300 - t.height) // 2))
    ban = ban.rotate(14, Image.BICUBIC, expand=True)
    im.alpha_composite(ban, ((S - ban.width) // 2, (S - ban.height) // 2))
    # grunge: knock random specks out of the ink
    n = np.random.default_rng(3).random((S // 4, S // 4))
    n = cv2.resize((n > 0.82).astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_NEAREST)
    n = cv2.GaussianBlur(n, (0, 0), 1.2)
    a = np.asarray(im.getchannel('A')).astype(np.float32) * (1 - n / 255 * 0.85)
    im.putalpha(Image.fromarray(a.astype(np.uint8)))
    return with_shadow(im.resize((size, size), Image.LANCZOS), 0.25, 12, 8)
STAMP = gift_stamp()

# ---------------------------------------------------------------- backgrounds
rng = np.random.default_rng(11)
GRAIN = [rng.normal(0, 5, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]

def grain(arr, i, amt=1.0):
    g = cv2.resize(GRAIN[i % 6], (W, H), interpolation=cv2.INTER_NEAREST)
    return np.clip(arr + g[..., None] * amt, 0, 255)

def radial(c_in, c_out, cx=.5, cy=.42):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.clip(np.sqrt(((xx / W - cx)) ** 2 + ((yy / H - cy) * 0.75) ** 2) / 0.8, 0, 1)[..., None]
    return np.array(c_in, np.float32) * (1 - d) + np.array(c_out, np.float32) * d

def halftone(arr, col=WHITE, op=0.16):
    im = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for gy in range(0, H, 34):
        for gx in range(0, W, 34):
            u = max(1 - math.hypot(gx / W, gy / H) / 0.55, 1 - math.hypot(1 - gx / W, 1 - gy / H) / 0.55)
            if u > 0:
                r = 2 + 8 * u
                d.ellipse((gx - r, gy - r, gx + r, gy + r), fill=col + (int(255 * op),))
    im.alpha_composite(ov)
    return np.asarray(im.convert('RGB')).astype(np.float32)

BG_PURPLE = halftone(radial((132, 100, 250), PURPLE_D))
BG_PINK = halftone(radial((255, 110, 180), (140, 52, 210), cy=.35))
BG_YELLOW = halftone(radial((255, 236, 120), (255, 190, 40)), INKP, 0.08)

SPARKS = [(150, 260, 26), (930, 420, 20), (120, 1500, 22), (960, 1650, 30), (880, 180, 16), (230, 1780, 14)]
def sparkles(cv, ts, col=WHITE):
    d = ImageDraw.Draw(cv)
    for n, (x, y, r) in enumerate(SPARKS):
        s = 0.5 + 0.5 * math.sin(ts * 5 + n * 1.7)
        if s > 0.15: d.polygon(star_pts(x, y, r * s), fill=col + (230,))

# ---------------------------------------------------------------- confetti
class Burst:
    def __init__(self, t0, x, y, n, seed, spread=70, vmin=900, vmax=1900):
        r = np.random.default_rng(seed)
        self.t0, self.x, self.y = t0, x, y
        ang = np.radians(-90 + r.uniform(-spread, spread, n))
        v = r.uniform(vmin, vmax, n)
        self.vx, self.vy = v * np.cos(ang), v * np.sin(ang)
        self.col = [CONF[j] for j in r.integers(0, len(CONF), n)]
        self.w = r.uniform(12, 24, n); self.h = r.uniform(7, 13, n)
        self.spin = r.uniform(-9, 9, n); self.flip = r.uniform(6, 14, n); self.ph = r.uniform(0, 6, n)
        self.round = r.random(n) < 0.25
    def draw(self, cv, ts, life=2.3):
        dt = ts - self.t0
        if dt < 0 or dt > life: return
        k, g = 1.7, 1500.0
        dx = self.vx * (1 - math.exp(-k * dt)) / k
        dy = self.vy * (1 - math.exp(-k * dt)) / k + 0.5 * g * dt * dt * 0.55
        sz = min(1.0, (life - dt) / 0.5)
        d = ImageDraw.Draw(cv)
        for j in range(len(self.vx)):
            cx, cy = self.x + dx[j] + 18 * math.sin(dt * 4 + self.ph[j]), self.y + dy[j]
            if not (-30 < cx < W + 30 and -30 < cy < H + 30): continue
            if self.round[j]:
                r = 6 * sz; d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self.col[j] + (255,)); continue
            a = self.spin[j] * dt + self.ph[j]
            w = self.w[j] * sz * abs(math.cos(self.flip[j] * dt + self.ph[j])) + 2
            h = self.h[j] * sz
            ca, sa = math.cos(a), math.sin(a)
            pts = [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in
                   ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
            d.polygon(pts, fill=self.col[j] + (255,))

BURSTS = [Burst(0.66, 540, 420, 70, 1), Burst(8.04, 540, 1000, 110, 2), Burst(13.88, 540, 1260, 80, 3),
          Burst(22.08, 540, 980, 100, 4), Burst(28.12, 540, 1000, 110, 5)]

# ---------------------------------------------------------------- media
def still(t):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', SRC, '-frames:v', '1', '-vf', GRADE,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)

def clip(name, off, dur, cx=0.5):
    """Decode a stock clip segment, crop to 9:16 around cx, scale to WxH at the edit fps."""
    vf = (f"crop='min(iw,ih*9/16)':ih:'(iw-min(iw,ih*9/16))*{cx}':0,scale={W}:{H}:flags=lanczos,"
          f"fps={FPS_N}/{FPS_D}")
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(off), '-i', MIX + name, '-t', str(dur + 0.2), '-vf', vf,
                          '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8)
    return a.reshape(-1, H, W, 3)

def card_img(arr, box, size=740, r=36, border=14):
    x0, y0, x1, y1 = box
    im = Image.fromarray(arr[y0:y1, x0:x1]).resize((size, size), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(3, 60, 2))
    m = Image.new('L', (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), r, fill=255)
    im.putalpha(m)
    S = size + 2 * border
    out = Image.new('RGBA', (S + 40, S + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((26, 30, 26 + S, 30 + S), r + border, fill=INKP + (255,))
    d.rounded_rectangle((20, 20, 20 + S, 20 + S), r + border, fill=WHITE + (255,))
    out.alpha_composite(im, (20 + border, 20 + border))
    return with_shadow(out, 0.3, 18, 12)

S36, S19 = still(3.6), still(19.0)
PROD = {
    '2way': card_img(S19, (0, 1060, 520, 1580)),
    'midbass': card_img(S36, (690, 740, 1010, 1060)),
    'sub': card_img(S19, (380, 1250, 910, 1780)),
}

# ---------------------------------------------------------------- edit plan (source seconds)
# A-roll: (start, end, zoom, [(appear, kind, text)])  kinds: lead | plate | key
A = [
    (0.60, 1.98, 1.00, [(0.90, 'lead', 'BARANG-BARANG AUDIO')]),
    (5.30, 7.32, 1.14, [(5.30, 'lead', 'DEKAT LUAR SANA'), (6.04, 'plate', 'MEMANG MAHAL')]),
    (9.30, 11.80, 1.00, [(9.36, 'lead', 'TURUN HARGA'), (10.42, 'plate', 'KAU-KAU'), (10.78, 'key', 'JATUH GILA')]),
    (15.24, 16.36, 1.14, [(15.24, 'lead', 'MACAM-MACAM'), (15.78, 'plate', 'BARANG AUDIO')]),
    (16.36, 18.48, 1.00, [(17.12, 'lead', 'KALAU HAMPA NAK'), (17.90, 'plate', 'BARANG')]),
    (18.48, 19.70, 1.20, [(18.48, 'lead', 'YANG KAU-KAU PUNYA'), (19.30, 'plate', 'MURAH')]),
    (19.70, 21.74, 1.10, [(19.70, 'lead', 'YANG TAK JUMPA'), (20.60, 'plate', 'DEKAT MANA-MANA')]),
    (23.10, 25.24, 1.00, [(23.10, 'lead', 'JADI JUMPA'), (24.00, 'plate', '11.11 NI YA')]),
    (26.40, 28.10, 1.14, [(26.40, 'lead', 'MASUK LIVE VIDEO'), (27.16, 'plate', 'ADA MURAH JUGA')]),
]
# B-roll: (start, end, kind, params)
B = [
    (1.98, 2.90, 'stock', dict(clip='hd_72.mp4', off=6.0, cx=0.8, caps=[(0.0, 'lead', 'BARANG-BARANG AUDIO'), (1.98, 'plate', 'DALAM KERETA')])),
    (2.90, 3.95, 'prod', dict(img='2way', idx='01', word='2 WAY', wt=3.68, rot=-3)),
    (3.95, 4.68, 'prod', dict(img='midbass', idx='02', word='MIDBASS', wt=4.40, rot=2.5)),
    (4.68, 5.30, 'prod', dict(img='sub', idx='03', word='SUBWOOFER', wt=4.78, rot=-2)),
    (7.32, 9.30, 'sale', dict(lt=7.32, bt=8.04)),
    (11.80, 15.24, 'list', dict(items=[(12.22, 'VOUCHER'), (13.16, 'DISCOUNT')], st=13.88, mt=14.46)),
    (21.74, 23.10, 'yellow', dict(lt=21.74, bt=22.08, tt=22.72)),
    (25.24, 26.40, 'stock', dict(clip='hd_41174.mp4', off=0.8, cx=0.5, caps=[(25.24, 'lead', 'MASUK LIVE VIDEO')])),
    (28.10, 28.90, 'end', {}),
]
CUTS = sorted({s for s, *_ in A + B} | {e for _, e, *_ in A + B})[1:-1]
LIVE_FROM = 25.54

CLIPS = {s: clip(p['clip'], p['off'], e - s, p['cx']) for s, e, k, p in B if k == 'stock'}
FIREWORKS = clip('hd_4064.mp4', 0.3, 9.30 - 7.32, 0.5)

# ---------------------------------------------------------------- shot renderers
def captions(cv, lines, ts):
    y = 1110
    for at, kind, txt in lines:
        if kind == 'lead': im, h = text(txt, ID + 'Inter-ExtraBold.otf', 56, WHITE, track=2), None
        elif kind == 'plate': im = pill(txt, ANTON, 104)
        else: im = text(txt, ANTON, 120, WHITE, track=2)
        h = im.height - (60 if kind == 'plate' else 50)
        cy = y + h / 2
        k = (ts - at) / 0.24
        if k > 0:
            if kind == 'plate': paste(cv, im, W / 2, cy, scale=0.8 + 0.2 * ease_back(k), alpha=min(k * 2, 1))
            else: paste(cv, im, W / 2, cy + 30 * (1 - ease_out(k)), alpha=ease_out(k))
            if kind == 'key':
                full = im.width - 120
                lw = int(full * ease_out((ts - at - 0.12) / 0.3))
                if lw > 0:
                    ImageDraw.Draw(cv).rounded_rectangle((W / 2 - full / 2, cy + h / 2 + 8, W / 2 - full / 2 + lw, cy + h / 2 + 22),
                                                         7, fill=PINK + (255,))
        y += h + 14

def hook(cv, ts):
    k = (ts - T0 - 0.04) / 0.34
    if k > 0: paste(cv, BADGE, W / 2, 400, scale=0.6 * (0.6 + 0.4 * ease_back(k)), alpha=min(1, k * 2.5),
                    rot=3 * math.sin(ts * 2.2))

def live_badge(cv, ts):
    k = (ts - LIVE_FROM) / 0.3
    if k <= 0: return
    b = Image.new('RGBA', (360, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(b)
    d.rounded_rectangle((16, 22, 346, 128), 52, fill=INKP + (255,))
    d.rounded_rectangle((10, 12, 340, 118), 52, fill=RED + (255,), outline=WHITE + (255,), width=5)
    pulse = 0.55 + 0.45 * abs(math.sin(ts * 4.0))
    d.ellipse((52, 47, 88, 83), fill=(255, 255, 255, int(255 * pulse)))
    b.alpha_composite(text('LIVE', ARCH, 56, WHITE, shadow=0, pad=2), (110, 33))
    paste(cv, b, W / 2, 240, scale=0.7 + 0.3 * ease_back(k), alpha=min(1, k * 2))

def bg(arr, i): return Image.fromarray(grain(arr, i, 0.6).astype(np.uint8)).convert('RGBA')

def r_prod(p, ts, s, e, i):
    cv = bg(BG_PURPLE, i)
    sparkles(cv, ts)
    k = (ts - s) / 0.32
    kb = 1.0 + 0.04 * (ts - s) / (e - s)
    paste(cv, PROD[p['img']], W / 2, 820 + 60 * (1 - ease_out(k)), scale=(0.9 + 0.1 * ease_back(k)) * kb,
          alpha=ease_out(k * 1.5), rot=p['rot'])
    paste(cv, pill('KHUSUSNYA', ARCH, 40), W / 2, 330, scale=0.8 + 0.2 * ease_back(k), alpha=ease_out(k))
    paste(cv, text(p['idx'] + ' / 03', ID + 'Inter-Bold.otf', 32, WHITE, track=4, shadow=0), W - 150, 150, alpha=ease_out(k))
    kw = (ts - p['wt']) / 0.22
    if kw > 0:
        paste(cv, text3d(p['word'], ARCH, 128, WHITE, PINK), W / 2, 1400 + 40 * (1 - ease_out(kw)),
              scale=0.85 + 0.15 * ease_back(kw), alpha=min(1, kw * 2))
    return cv

def r_sale(p, ts, s, e, i):
    base = grain(BG_PURPLE, i, 0.6)
    fw = FIREWORKS[min(len(FIREWORKS) - 1, int((ts - s) * FPS))].astype(np.float32)
    arr = 255 - (255 - base) * (255 - fw * 0.85) / 255          # screen blend
    cv = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    k = (ts - p['lt']) / 0.25
    kb = (ts - p['bt']) / 0.3
    if kb <= 0:
        paste(cv, text3d('TAPI', ARCH, 150, WHITE, PINK), W / 2, 800 + 30 * (1 - ease_out(k)), alpha=ease_out(k))
        for j in range(3):
            a = 0.35 + 0.65 * max(0, math.sin(ts * 9 - j))
            ImageDraw.Draw(cv).ellipse((W / 2 - 70 + j * 60 - 14, 1000, W / 2 - 70 + j * 60 + 14, 1028), fill=YELLOW + (int(255 * a),))
    else:
        paste(cv, text('TAPI', ARCH, 64, WHITE, track=10), W / 2, 520, alpha=1)
        shake = 14 * math.exp(-(ts - p['bt']) * 12) * math.sin((ts - p['bt']) * 80)
        paste(cv, BADGE, W / 2 + shake, 1000, scale=0.95 * (0.5 + 0.5 * ease_back(kb)), alpha=min(1, kb * 3),
              rot=2 * math.sin(ts * 2.5))
    return cv

def r_list(p, ts, s, e, i):
    cv = bg(BG_PINK, i)
    sparkles(cv, ts, YELLOW)
    k = (ts - s) / 0.25
    paste(cv, pill('ADA', ARCH, 54), W / 2, 420, scale=0.8 + 0.2 * ease_back(k), alpha=ease_out(k))
    for n, (at, word) in enumerate(p['items']):
        kk = (ts - at) / 0.28
        if kk <= 0: continue
        y = 700 + n * 210
        x_off = -140 * (1 - ease_out(kk))
        tick = Image.new('RGBA', (120, 120), (0, 0, 0, 0))
        td = ImageDraw.Draw(tick)
        td.ellipse((6, 8, 114, 116), fill=INKP + (255,)); td.ellipse((2, 2, 108, 108), fill=YELLOW + (255,), outline=INKP + (255,), width=5)
        td.line([(30, 56), (48, 74), (80, 38)], fill=INKP + (255,), width=12, joint='curve')
        paste(cv, tick, 190 + x_off, y, scale=0.6 + 0.4 * ease_back(kk), alpha=min(1, kk * 2))
        wi = text3d(word, ARCH, 118, WHITE, PURPLE_D, depth=12, stroke=8)
        paste(cv, wi, 270 + wi.width / 2 + x_off, y, alpha=min(1, kk * 2))
    ks = (ts - p['st']) / 0.22
    if ks > 0:
        paste(cv, STAMP, W / 2, 1260, scale=1.12 + 0.7 * (1 - ease_out(ks)), alpha=min(1, ks * 2), rot=-6)
    km = (ts - p['mt']) / 0.25
    if km > 0:
        paste(cv, pill('MELAMBAK GILA!', ARCH, 60, INKP, YELLOW), W / 2, 1730, scale=0.7 + 0.3 * ease_back(km), alpha=min(1, km * 2), rot=3)
    return cv

def r_yellow(p, ts, s, e, i):
    cv = bg(BG_YELLOW, i)
    sparkles(cv, ts, WHITE)
    k = (ts - p['lt']) / 0.25
    paste(cv, text3d('HANYA', ARCH, 110, WHITE, PINK), W / 2, 600 + 30 * (1 - ease_out(k)), alpha=ease_out(k))
    kb = (ts - p['bt']) / 0.25
    if kb > 0:
        shake = 12 * math.exp(-(ts - p['bt']) * 12) * math.sin((ts - p['bt']) * 80)
        paste(cv, text3d('11.11', ARCH, 300, PINK, PURPLE, depth=20, stroke=12), W / 2 + shake, 960,
              scale=1 + 0.45 * (1 - ease_out(kb)), alpha=min(1, kb * 2))
    kt = (ts - p['tt']) / 0.25
    if kt > 0:
        paste(cv, text3d('SAJA!', ARCH, 150, WHITE, PURPLE), W / 2, 1320, scale=0.7 + 0.3 * ease_back(kt), alpha=min(1, kt * 2), rot=-3)
    return cv

def r_stock(p, ts, s, e, i):
    fr = CLIPS[s]
    cv = Image.fromarray(fr[min(len(fr) - 1, int((ts - s) * FPS))]).convert('RGBA')
    if ts >= LIVE_FROM: live_badge(cv, ts)
    captions(cv, [(at if at else s, kd, tx) for at, kd, tx in p['caps']], ts)
    return cv

def r_end(ts, s, e, i):
    cv = bg(BG_PURPLE, i)
    sparkles(cv, ts, YELLOW)
    k = (ts - s) / 0.3
    paste(cv, text3d('JUMPA', ARCH, 110, WHITE, PINK), W / 2, 520, alpha=ease_out(k))
    paste(cv, BADGE, W / 2, 960, scale=0.95 * (0.6 + 0.4 * ease_back(k)), alpha=min(1, k * 2.5), rot=2 * math.sin(ts * 2.5))
    paste(cv, pill('AUDIO KERETA  ·  LIVE', ARCH, 46, INKP, WHITE), W / 2, 1420,
          scale=0.8 + 0.2 * ease_back((ts - s - 0.2) / 0.3), alpha=ease_out((ts - s - 0.2) / 0.3))
    return cv

def a_roll(base, z):
    if z == 1.0: return base
    cx, cy = W / 2, 820
    M = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
    return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

def punch(arr, since):
    """Zoom-blur punch for the first frames after a cut."""
    n = 5
    k = since * FPS
    if k >= n: return arr
    s = 1 + 0.07 * (1 - k / n)
    acc = np.zeros_like(arr, np.float32)
    for sc in (s, s * 1.025, s * 1.05):
        M = np.float32([[sc, 0, W / 2 - sc * W / 2], [0, sc, H / 2 - sc * H / 2]])
        acc += cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (acc / 3).astype(np.uint8)

R = {'prod': r_prod, 'sale': r_sale, 'list': r_list, 'yellow': r_yellow, 'stock': r_stock}

def frame(base, ts, i):
    for s, e, kind, p in B:
        if s <= ts < e:
            cv = r_end(ts, s, e, i) if kind == 'end' else R[kind](p, ts, s, e, i)
            break
    else:
        shot = next((a for a in A if a[0] <= ts < a[1]), A[-1])
        cv = Image.fromarray(a_roll(base, shot[2])).convert('RGBA')
        if ts < 1.98: hook(cv, ts)
        if ts >= LIVE_FROM: live_badge(cv, ts)
        captions(cv, shot[3], ts)
    for b in BURSTS: b.draw(cv, ts)
    out = np.asarray(cv.convert('RGB'))
    last = max([c for c in CUTS if c <= ts], default=None)
    if last is not None: out = punch(out, ts - last)
    return out

# ---------------------------------------------------------------- main
if os.environ.get('TEST'):
    ts_list = [float(x) for x in os.environ['TEST'].split(',')]
    tiles = [Image.fromarray(frame(still(t), t, n)).resize((360, 640)) for n, t in enumerate(ts_list)]
    cols = min(6, len(tiles)); rows = math.ceil(len(tiles) / cols)
    sheet = Image.new('RGB', (360 * cols, 640 * rows))
    for n, im in enumerate(tiles): sheet.paste(im, ((n % cols) * 360, (n // cols) * 640))
    sheet.save(HERE + '/work/test.jpg', quality=88)
    sys.exit()

dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(T0), '-i', SRC, '-t', str(T1 - T0), '-vf', GRADE,
                        '-r', f'{FPS_N}/{FPS_D}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                        '-r', f'{FPS_N}/{FPS_D}', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
i = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    ts = T0 + i / FPS
    enc.stdin.write(frame(np.frombuffer(buf, np.uint8).reshape(H, W, 3), ts, i).tobytes())
    i += 1
    if i % 120 == 0: print(f'{i} frames ({ts:.1f}s)', flush=True)
enc.stdin.close(); enc.wait(); dec.wait()
print('done', i, 'frames')
