"""PROMO 11.11 — corporate-but-festive commercial re-edit of the car-audio talking-head promo (IMG_5148.MOV).

A-roll = original footage (HLG HDR tone-mapped to SDR to match the phone's look), punch-in jump cuts,
chest captions. B-roll = product cards cut from the footage, Mixkit stock clips (free licence) and clean
brand-red / off-white 11.11 cards with gold accents and light gold confetti.

Usage: python3 render.py SRC.MOV OUT_VIDEO_ONLY.mp4      (env TEST=t1,t2 -> work/test.jpg, source times)
Needs work/mixkit/hd_{72,41174,4064}.mp4.
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

BRAND = (200, 16, 46)
BRAND_D = (128, 8, 26)
GOLD = (212, 175, 55)
GOLD_L = (240, 214, 130)
INK = (17, 24, 39)
GREY = (110, 116, 128)
WHITE = (255, 255, 255)
CONF = [GOLD, GOLD, GOLD_L, WHITE, BRAND]

ID = '/usr/share/fonts/opentype/inter/'
BLACK, XB, BOLD, SEMI = ID + 'Inter-Black.otf', ID + 'Inter-ExtraBold.otf', ID + 'Inter-Bold.otf', ID + 'Inter-SemiBold.otf'
MIX = HERE + '/work/mixkit/'
# iPhone HLG -> SDR bt709, tuned to match how the clip looks on the phone
GRADE = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
         'zscale=t=bt709:m=bt709:r=tv,format=yuv420p,eq=gamma=0.8:brightness=-0.06:saturation=0.75')

_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc: _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_back(x):
    x = min(max(x, 0), 1); c1 = 1.2; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

# ---------------------------------------------------------------- text helpers
_tc = {}
def text(txt, path, size, fill=WHITE, track=0, shadow=0.0, pad=40):
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

def plate(txt, size=92, fg=INK, bg=WHITE, accent=BRAND, padx=34, pady=18):
    """Clean white caption plate with a brand-red accent bar on the left."""
    k = ('plate', txt, size, fg, bg, accent)
    if k in _tc: return _tc[k]
    t = text(txt, BLACK, size, fg, pad=4)
    w, h = t.width + 2 * padx + 14, t.height + 2 * pady
    im = Image.new('RGBA', (w + 60, h + 60), (0, 0, 0, 0))
    sh = Image.new('L', im.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((30, 40, 30 + w, 40 + h), 12, fill=110)
    im.putalpha(sh.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 30, 30 + w, 30 + h), 12, fill=bg + (255,))
    d.rounded_rectangle((30, 30, 44, 30 + h), 6, fill=accent + (255,))
    im.alpha_composite(t, (30 + padx + 14, 30 + pady))
    _tc[k] = im
    return im

def tag(txt, size=34, fg=WHITE, bg=BRAND, track=8):
    """Small letter-spaced label tag."""
    k = ('tag', txt, size, fg, bg)
    if k in _tc: return _tc[k]
    t = text(txt, BOLD, size, fg, track=track, pad=2)
    w, h = t.width + 48, t.height + 26
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), 8, fill=bg + (255,))
    im.alpha_composite(t, (24, 13))
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

def rule(cv, cx, y, full, k, col=GOLD, th=6):
    lw = int(full * ease_out(k))
    if lw > 0: ImageDraw.Draw(cv).rectangle((cx - lw / 2, y, cx + lw / 2, y + th), fill=col + (255,))

# ---------------------------------------------------------------- brand marks
def radial_small(size, c_in, c_out):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.clip(np.sqrt((xx / w - .5) ** 2 + (yy / h - .4) ** 2) / 0.75, 0, 1)[..., None]
    return (np.array(c_in, np.float32) * (1 - d) + np.array(c_out, np.float32) * d).astype(np.uint8)

def sale_mark():
    """Corporate 11.11 SALE lock-up: big white numerals, gold rules, letter-spaced SALE."""
    im = Image.new('RGBA', (900, 480), (0, 0, 0, 0))
    n = text('11.11', BLACK, 250, WHITE, track=-4, pad=6)
    im.alpha_composite(n, ((900 - n.width) // 2, 0))
    s = text('SALE', BOLD, 64, GOLD_L, track=34, pad=4)
    d = ImageDraw.Draw(im)
    y = n.height + 30
    d.rectangle((90, y + s.height // 2 - 2, 450 - s.width // 2 - 30, y + s.height // 2 + 2), fill=GOLD + (255,))
    d.rectangle((450 + s.width // 2 + 30, y + s.height // 2 - 2, 810, y + s.height // 2 + 2), fill=GOLD + (255,))
    im.alpha_composite(s, ((900 - s.width) // 2 + 17, y))
    im = im.crop(im.getbbox())
    return with_shadow(im, 0.35, 16, 8)
MARK = sale_mark()

def hook_badge():
    """Compact red card with the 11.11 SALE mark for the opening shot."""
    m = MARK.resize((int(MARK.width * 0.62), int(MARK.height * 0.62)), Image.LANCZOS)
    w, h = m.width + 110, m.height + 90
    im = Image.new('RGBA', (w + 60, h + 60), (0, 0, 0, 0))
    sh = Image.new('L', im.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((30, 42, 30 + w, 42 + h), 26, fill=130)
    im.putalpha(sh.filter(ImageFilter.GaussianBlur(18)))
    card = Image.fromarray(radial_small((w, h), (220, 28, 56), BRAND_D)).convert('RGBA')
    mask = Image.new('L', (w, h), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), 26, fill=255)
    card.putalpha(mask)
    im.alpha_composite(card, (30, 30))
    ImageDraw.Draw(im).rounded_rectangle((30, 30, 30 + w - 1, 30 + h - 1), 26, outline=GOLD + (255,), width=4)
    im.alpha_composite(m, (30 + (w - m.width) // 2, 30 + (h - m.height) // 2))
    return im
HOOK = hook_badge()

def gift_icon(size=96, col=BRAND):
    im = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = col + (255,)
    d.rounded_rectangle((14, 44, 86, 92), 6, outline=c, width=7)
    d.rounded_rectangle((8, 30, 92, 46), 5, fill=c)
    d.rectangle((46, 30, 54, 92), fill=c)
    d.ellipse((24, 10, 50, 32), outline=c, width=6); d.ellipse((50, 10, 76, 32), outline=c, width=6)
    return im.resize((size, size), Image.LANCZOS)

# ---------------------------------------------------------------- backgrounds
rng = np.random.default_rng(11)
GRAIN = [rng.normal(0, 3.5, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]

def grain(arr, i, amt=1.0):
    g = cv2.resize(GRAIN[i % 6], (W, H), interpolation=cv2.INTER_NEAREST)
    return np.clip(arr + g[..., None] * amt, 0, 255)

def radial(c_in, c_out, cx=.5, cy=.42):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.clip(np.sqrt(((xx / W - cx)) ** 2 + ((yy / H - cy) * 0.75) ** 2) / 0.8, 0, 1)[..., None]
    return np.array(c_in, np.float32) * (1 - d) + np.array(c_out, np.float32) * d

def corner_lines(arr, col, op):
    """Thin diagonal pinstripes in two corners — a quiet corporate texture."""
    im = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for j in range(0, 520, 26):
        d.line([(W - 520 + j, 0), (W, 520 - j)], fill=col + (int(255 * op),), width=2)
        d.line([(0, H - 520 + j), (520 - j, H)], fill=col + (int(255 * op),), width=2)
    im.alpha_composite(ov)
    return np.asarray(im.convert('RGB')).astype(np.float32)

BG_RED = corner_lines(radial((222, 30, 58), BRAND_D), GOLD, 0.18)
BG_PAPER = corner_lines(radial((255, 255, 253), (232, 229, 224)), BRAND, 0.10)

def star_pts(cx, cy, r, inner=0.28):
    pts = []
    for j in range(8):
        a = -math.pi / 2 + j * math.pi / 4
        rr = r if j % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts

GLINTS = [(170, 420, 18), (920, 600, 14), (150, 1420, 14), (930, 1500, 20)]
def glints(cv, ts, col=GOLD_L):
    d = ImageDraw.Draw(cv)
    for n, (x, y, r) in enumerate(GLINTS):
        s = 0.5 + 0.5 * math.sin(ts * 3.5 + n * 1.9)
        if s > 0.2: d.polygon(star_pts(x, y, r * s), fill=col + (220,))

# ---------------------------------------------------------------- confetti (light, gold)
class Burst:
    def __init__(self, t0, x, y, n, seed, spread=65, vmin=800, vmax=1600):
        r = np.random.default_rng(seed)
        self.t0, self.x, self.y = t0, x, y
        ang = np.radians(-90 + r.uniform(-spread, spread, n))
        v = r.uniform(vmin, vmax, n)
        self.vx, self.vy = v * np.cos(ang), v * np.sin(ang)
        self.col = [CONF[j] for j in r.integers(0, len(CONF), n)]
        self.w = r.uniform(8, 16, n); self.h = r.uniform(4, 7, n)
        self.spin = r.uniform(-8, 8, n); self.flip = r.uniform(5, 12, n); self.ph = r.uniform(0, 6, n)
    def draw(self, cv, ts, life=2.2):
        dt = ts - self.t0
        if dt < 0 or dt > life: return
        k, g = 1.8, 800.0
        dx = self.vx * (1 - math.exp(-k * dt)) / k
        dy = self.vy * (1 - math.exp(-k * dt)) / k + 0.5 * g * dt * dt
        sz = min(1.0, (life - dt) / 0.6)
        d = ImageDraw.Draw(cv)
        for j in range(len(self.vx)):
            cx, cy = self.x + dx[j] + 16 * math.sin(dt * 4 + self.ph[j]), self.y + dy[j]
            if not (-30 < cx < W + 30 and -30 < cy < H + 30): continue
            a = self.spin[j] * dt + self.ph[j]
            w = self.w[j] * sz * abs(math.cos(self.flip[j] * dt + self.ph[j])) + 1.5
            h = self.h[j] * sz
            ca, sa = math.cos(a), math.sin(a)
            pts = [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in
                   ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
            d.polygon(pts, fill=self.col[j] + (255,))

BURSTS = [Burst(0.70, 540, 430, 45, 1), Burst(8.04, 540, 960, 70, 2), Burst(13.88, 540, 1240, 45, 3),
          Burst(22.08, 540, 960, 60, 4), Burst(28.12, 540, 960, 70, 5)]

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
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)

def card_img(arr, box, size=720, r=28):
    x0, y0, x1, y1 = box
    im = Image.fromarray(arr[y0:y1, x0:x1]).resize((size, size), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(3, 60, 2))
    m = Image.new('L', (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), r, fill=255)
    im.putalpha(m)
    out = Image.new('RGBA', (size + 100, size + 100), (0, 0, 0, 0))
    sh = Image.new('L', out.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((50, 66, 50 + size, 66 + size), r, fill=120)
    out.putalpha(sh.filter(ImageFilter.GaussianBlur(24)))
    out.alpha_composite(im, (50, 50))
    return out

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
    (2.90, 3.95, 'prod', dict(img='2way', idx='01', word='2 WAY', wt=3.68)),
    (3.95, 4.68, 'prod', dict(img='midbass', idx='02', word='MIDBASS', wt=4.40)),
    (4.68, 5.30, 'prod', dict(img='sub', idx='03', word='SUBWOOFER', wt=4.78)),
    (7.32, 9.30, 'sale', dict(lt=7.32, bt=8.04)),
    (11.80, 15.24, 'list', dict(items=[(12.22, 'VOUCHER'), (13.16, 'DISCOUNT'), (13.88, 'FREE GIFT')], mt=14.46)),
    (21.74, 23.10, 'only', dict(lt=21.74, bt=22.08, tt=22.72)),
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
        if kind == 'lead': im = text(txt, XB, 54, WHITE, track=3, shadow=0.6)
        elif kind == 'plate': im = plate(txt)
        else: im = text(txt, BLACK, 112, WHITE, track=1, shadow=0.6)
        h = im.height - (60 if kind == 'plate' else 80)
        cy = y + h / 2
        k = (ts - at) / 0.26
        if k > 0:
            if kind == 'plate': paste(cv, im, W / 2, cy + 20 * (1 - ease_out(k)), scale=0.94 + 0.06 * ease_out(k), alpha=ease_out(k * 1.5))
            else: paste(cv, im, W / 2, cy + 26 * (1 - ease_out(k)), alpha=ease_out(k))
            if kind == 'key': rule(cv, W / 2, cy + h / 2 + 12, im.width - 140, (ts - at - 0.12) / 0.35, GOLD, 8)
        y += h + 16

def hook(cv, ts):
    k = (ts - T0 - 0.04) / 0.4
    if k > 0: paste(cv, HOOK, W / 2, 380 + 30 * (1 - ease_out(k)), scale=0.92 + 0.08 * ease_back(k), alpha=ease_out(k * 1.4))

def live_badge(cv, ts):
    k = (ts - LIVE_FROM) / 0.3
    if k <= 0: return
    b = Image.new('RGBA', (300, 110), (0, 0, 0, 0))
    d = ImageDraw.Draw(b)
    d.rounded_rectangle((4, 4, 296, 106), 18, fill=BRAND + (255,))
    pulse = 0.55 + 0.45 * abs(math.sin(ts * 4.0))
    d.ellipse((40, 37, 76, 73), fill=(255, 255, 255, int(255 * pulse)))
    b.alpha_composite(text('LIVE', BLACK, 52, WHITE, track=6, pad=2), (98, 26))
    paste(cv, with_shadow(b, 0.35, 12, 6), W / 2, 240, scale=0.9 + 0.1 * ease_back(k), alpha=ease_out(k * 1.5))

def bg(arr, i): return Image.fromarray(grain(arr, i, 0.6).astype(np.uint8)).convert('RGBA')

def r_prod(p, ts, s, e, i):
    cv = bg(BG_PAPER, i)
    k = (ts - s) / 0.35
    kb = 1.0 + 0.035 * (ts - s) / (e - s)
    paste(cv, PROD[p['img']], W / 2, 860 + 50 * (1 - ease_out(k)), scale=(0.95 + 0.05 * ease_out(k)) * kb, alpha=ease_out(k * 1.5))
    paste(cv, tag('KHUSUSNYA'), W / 2, 330, alpha=ease_out(k))
    paste(cv, text(p['idx'] + ' — 03', SEMI, 32, GREY, track=6), W - 160, 150, alpha=ease_out(k))
    kw = (ts - p['wt']) / 0.26
    if kw > 0:
        paste(cv, text(p['word'], BLACK, 132, INK, track=2), W / 2, 1420 + 30 * (1 - ease_out(kw)), alpha=ease_out(kw))
        rule(cv, W / 2, 1520, 180, (ts - p['wt'] - 0.1) / 0.35, BRAND, 8)
    return cv

def r_sale(p, ts, s, e, i):
    base = grain(BG_RED, i, 0.6)
    fw = FIREWORKS[min(len(FIREWORKS) - 1, int((ts - s) * FPS))].astype(np.float32)
    fw = fw.mean(2, keepdims=True) * np.array(GOLD_L, np.float32) / 255   # recolour fireworks to gold
    arr = 255 - (255 - base) * (255 - fw * 0.6) / 255                     # screen blend
    cv = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    glints(cv, ts)
    k = (ts - p['lt']) / 0.3
    kb = (ts - p['bt']) / 0.35
    paste(cv, text('TAPI', BOLD, 54, WHITE, track=22), W / 2, 560 if kb > 0 else 820, alpha=ease_out(k))
    if kb <= 0:
        rule(cv, W / 2, 880, 120, k, GOLD, 4)
    else:
        paste(cv, MARK, W / 2, 960, scale=1.0 + 0.12 * (1 - ease_out(kb)), alpha=ease_out(kb * 1.6))
    return cv

def r_list(p, ts, s, e, i):
    cv = bg(BG_PAPER, i)
    k = (ts - s) / 0.3
    paste(cv, tag('ADA'), W / 2, 520, alpha=ease_out(k))
    for n, (at, word) in enumerate(p['items']):
        kk = (ts - at) / 0.32
        if kk <= 0: continue
        y = 780 + n * 230
        hi = word == 'FREE GIFT'
        row = Image.new('RGBA', (860, 180), (0, 0, 0, 0))
        rd = ImageDraw.Draw(row)
        rd.rounded_rectangle((0, 0, 859, 179), 20, fill=(BRAND if hi else WHITE) + (255,))
        if hi: row.alpha_composite(gift_icon(96, WHITE), (50, 42))
        else:
            rd.rounded_rectangle((0, 0, 859, 179), 20, outline=(225, 222, 216, 255), width=3)
            rd.ellipse((50, 42, 146, 138), fill=BRAND + (255,))
            rd.line([(74, 92), (92, 110), (124, 72)], fill=WHITE + (255,), width=11, joint='curve')
        t = text(word, BLACK, 96, WHITE if hi else INK, track=1, pad=2)
        row.alpha_composite(t, (190, (180 - t.height) // 2))
        paste(cv, with_shadow(row, 0.18, 18, 10), W / 2 + 60 * (1 - ease_out(kk)), y, alpha=ease_out(kk * 1.5))
    km = (ts - p['mt']) / 0.3
    if km > 0:
        paste(cv, tag('MELAMBAK GILA!', 54, INK, GOLD_L, 6), W / 2, 1520, scale=0.94 + 0.06 * ease_back(km), alpha=ease_out(km * 1.5))
    return cv

def r_only(p, ts, s, e, i):
    cv = bg(BG_PAPER, i)
    k = (ts - p['lt']) / 0.3
    paste(cv, text('HANYA', BOLD, 60, INK, track=24), W / 2, 640 + 24 * (1 - ease_out(k)), alpha=ease_out(k))
    kb = (ts - p['bt']) / 0.32
    if kb > 0:
        paste(cv, text('11.11', BLACK, 300, BRAND, track=-6), W / 2, 930, scale=1 + 0.1 * (1 - ease_out(kb)), alpha=ease_out(kb * 1.6))
        rule(cv, W / 2, 1110, 420, (ts - p['bt'] - 0.1) / 0.4, GOLD, 6)
    kt = (ts - p['tt']) / 0.3
    if kt > 0:
        paste(cv, text('SAJA!', BLACK, 110, INK, track=4), W / 2, 1240 + 24 * (1 - ease_out(kt)), alpha=ease_out(kt))
    return cv

def r_stock(p, ts, s, e, i):
    fr = CLIPS[s]
    cv = Image.fromarray(fr[min(len(fr) - 1, int((ts - s) * FPS))]).convert('RGBA')
    if ts >= LIVE_FROM: live_badge(cv, ts)
    captions(cv, [(at if at else s, kd, tx) for at, kd, tx in p['caps']], ts)
    return cv

def r_end(ts, s, e, i):
    cv = bg(BG_RED, i)
    glints(cv, ts)
    k = (ts - s) / 0.35
    paste(cv, text('JUMPA', BOLD, 54, WHITE, track=24), W / 2, 640, alpha=ease_out(k))
    paste(cv, MARK, W / 2, 940, scale=1.0 + 0.1 * (1 - ease_out(k)), alpha=ease_out(k * 1.6))
    paste(cv, text('AUDIO KERETA   ·   LIVE', SEMI, 36, GOLD_L, track=10), W / 2, 1260, alpha=ease_out((ts - s - 0.2) / 0.3))
    return cv

def a_roll(base, z):
    if z == 1.0: return base
    cx, cy = W / 2, 820
    M = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
    return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

def punch(arr, since):
    """Short zoom-blur punch for the first frames after a cut."""
    n = 4
    k = since * FPS
    if k >= n: return arr
    s = 1 + 0.05 * (1 - k / n)
    acc = np.zeros_like(arr, np.float32)
    for sc in (s, s * 1.02, s * 1.04):
        M = np.float32([[sc, 0, W / 2 - sc * W / 2], [0, sc, H / 2 - sc * H / 2]])
        acc += cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (acc / 3).astype(np.uint8)

R = {'prod': r_prod, 'sale': r_sale, 'list': r_list, 'only': r_only, 'stock': r_stock}

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
                        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                        '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
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
