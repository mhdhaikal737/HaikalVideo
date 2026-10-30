"""PROMO 11.11 — commercial re-edit of the car-audio talking-head promo (IMG_5148.MOV).

A-roll = graded original footage with punch-in jump cuts and chest captions.
B-roll = product cards cut from the footage + designed 11.11 / offer / end cards.

Usage: python3 render.py SRC.MOV OUT_VIDEO_ONLY.mp4  (needs work/fonts/anton.ttf, archivo-black.ttf from @fontsource)      (env TEST=t1,t2 -> work/test.jpg, source times)
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

RED = (227, 38, 46)
WHITE = (255, 255, 255)
INK = (16, 16, 18)
GREY = (170, 170, 176)

ANTON = HERE + '/work/fonts/anton.ttf'
ARCH = HERE + '/work/fonts/archivo-black.ttf'
ID = '/usr/share/fonts/opentype/inter/'
GRADE = ('eq=contrast=1.07:brightness=-0.012:saturation=1.10,'
         'colorbalance=rs=.025:gs=.005:bs=-.02:rh=.02:bh=-.025,'
         'unsharp=5:5:0.45,vignette=angle=PI/5')

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
    if shadow:
        a = im.getchannel('A').filter(ImageFilter.GaussianBlur(10))
        sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
        sh.putalpha(a.point(lambda v: int(v * shadow)))
        base = Image.new('RGBA', im.size, (0, 0, 0, 0))
        base.alpha_composite(sh, (0, 6) if im.height > 6 else (0, 0))
        base.alpha_composite(im)
        im = base
    _tc[k] = im
    return im

def plate(txt, path, size, fg=INK, bg=WHITE, padx=30, pady=16, r=14):
    k = ('plate', txt, path, size, fg, bg)
    if k in _tc: return _tc[k]
    t = text(txt, path, size, fg, shadow=0, pad=4)
    w, h = t.width + 2 * padx, t.height + 2 * pady
    im = Image.new('RGBA', (w + 60, h + 60), (0, 0, 0, 0))
    sh = Image.new('L', im.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((30, 38, 30 + w, 38 + h), r, fill=120)
    im.putalpha(sh.filter(ImageFilter.GaussianBlur(12)))
    ImageDraw.Draw(im).rounded_rectangle((30, 30, 30 + w, 30 + h), r, fill=bg + (255,))
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

# ---------------------------------------------------------------- backgrounds
rng = np.random.default_rng(11)
GRAIN = [rng.normal(0, 7, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]

def grain(arr, i, amt=1.0):
    g = cv2.resize(GRAIN[i % 6], (W, H), interpolation=cv2.INTER_NEAREST)
    return np.clip(arr + g[..., None] * amt, 0, 255)

def radial(c_in, c_out, cx=.5, cy=.42, glow=None):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx / W - cx) * 1.0) ** 2 + ((yy / H - cy) * 0.75) ** 2) / 0.75
    d = np.clip(d, 0, 1)[..., None]
    a = np.array(c_in, np.float32) * (1 - d) + np.array(c_out, np.float32) * d
    if glow:
        gx, gy, gr, gc, ga = glow
        g = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * gr ** 2)))[..., None]
        a = a * (1 - g * ga) + np.array(gc, np.float32) * g * ga
    return a

BG_DARK = radial((30, 30, 34), (8, 8, 9), glow=(160, 1700, 420, RED, .35))
BG_RED = radial((196, 26, 34), (70, 6, 10), cy=.45)
BG_PAPER = radial((246, 244, 240), (214, 210, 204))

# ---------------------------------------------------------------- product stills (graded)
def still(t):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', SRC, '-frames:v', '1', '-vf', GRADE,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)

def card_img(arr, box, size=760, r=44):
    x0, y0, x1, y1 = box
    im = Image.fromarray(arr[y0:y1, x0:x1]).resize((size, size), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(3, 60, 2))
    m = Image.new('L', (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), r, fill=255)
    out = Image.new('RGBA', (size + 120, size + 120), (0, 0, 0, 0))
    sh = Image.new('L', out.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((60, 78, 60 + size, 78 + size), r, fill=170)
    out.putalpha(sh.filter(ImageFilter.GaussianBlur(26)))
    im.putalpha(m)
    out.alpha_composite(im, (60, 60))
    ImageDraw.Draw(out).rounded_rectangle((60, 60, 60 + size - 1, 60 + size - 1), r, outline=(255, 255, 255, 70), width=3)
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
    (0.60, 2.90, 1.00, [(0.90, 'lead', 'BARANG-BARANG AUDIO'), (1.98, 'plate', 'DALAM KERETA')]),
    (5.30, 7.32, 1.14, [(5.30, 'lead', 'DEKAT LUAR SANA'), (6.04, 'plate', 'MEMANG MAHAL')]),
    (9.30, 11.80, 1.00, [(9.36, 'lead', 'TURUN HARGA'), (10.42, 'plate', 'KAU-KAU'), (10.78, 'key', 'JATUH GILA')]),
    (15.24, 16.36, 1.14, [(15.24, 'lead', 'MACAM-MACAM'), (15.78, 'plate', 'BARANG AUDIO')]),
    (16.36, 18.48, 1.00, [(17.12, 'lead', 'KALAU HAMPA NAK'), (17.90, 'plate', 'BARANG')]),
    (18.48, 19.70, 1.20, [(18.48, 'lead', 'YANG KAU-KAU PUNYA'), (19.30, 'plate', 'MURAH')]),
    (19.70, 21.74, 1.10, [(19.70, 'lead', 'YANG TAK JUMPA'), (20.60, 'plate', 'DEKAT MANA-MANA')]),
    (23.10, 25.24, 1.00, [(23.10, 'lead', 'JADI JUMPA'), (24.00, 'plate', '11.11 NI YA')]),
    (25.24, 28.10, 1.14, [(25.24, 'lead', 'MASUK LIVE VIDEO'), (27.16, 'plate', 'ADA MURAH JUGA')]),
]
# B-roll: (start, end, kind, params)
B = [
    (2.90, 3.95, 'prod', dict(img='2way', idx='01', word='2 WAY', wt=3.68, label='KHUSUSNYA')),
    (3.95, 4.68, 'prod', dict(img='midbass', idx='02', word='MIDBASS', wt=4.40, label='KHUSUSNYA')),
    (4.68, 5.30, 'prod', dict(img='sub', idx='03', word='SUBWOOFER', wt=4.78, label='KHUSUSNYA')),
    (7.32, 9.30, 'sale', dict(lead='TAPI', lt=7.32, big='11.11', bt=8.04, tail='', tt=99)),
    (11.80, 15.24, 'list', dict(items=[(12.22, 'VOUCHER'), (13.16, 'DISCOUNT'), (13.88, 'FREE GIFT')],
                                 stamp='MELAMBAK GILA!', st=14.46)),
    (21.74, 23.10, 'paper', dict(lead='HANYA', lt=21.74, big='11.11', bt=22.08, tail='SAJA!', tt=22.72)),
    (28.10, 28.90, 'end', {}),
]
CUTS = sorted({s for s, *_ in A + B} | {e for _, e, *_ in A + B}) [1:-1]
HOOK_END = 2.90
LIVE_FROM = 25.54

# ---------------------------------------------------------------- shot renderers
def captions(cv, lines, ts):
    imgs = []
    for at, kind, txt in lines:
        if kind == 'lead': im = text(txt, ID + 'Inter-ExtraBold.otf', 56, WHITE, track=2)
        elif kind == 'plate': im = plate(txt, ANTON, 112)
        else: im = text(txt, ANTON, 120, WHITE, track=2)
        imgs.append((at, kind, im))
    y = 1110
    for at, kind, im in imgs:
        h = im.height - (60 if kind == 'plate' else 50)
        cy = y + h / 2
        k = (ts - at) / 0.24
        if k > 0:
            if kind == 'plate': paste(cv, im, W / 2, cy, scale=0.82 + 0.18 * ease_back(k), alpha=min(k * 2, 1))
            else: paste(cv, im, W / 2, cy + 30 * (1 - ease_out(k)), alpha=ease_out(k))
            if kind == 'key':
                lw = int((im.width - 120) * ease_out((ts - at - 0.12) / 0.3))
                if lw > 0:
                    ImageDraw.Draw(cv).rectangle((W / 2 - (im.width - 120) / 2, cy + h / 2 + 8,
                                                  W / 2 - (im.width - 120) / 2 + lw, cy + h / 2 + 18), fill=RED + (255,))
        y += h + 14

def top_shade(cv, strength=170, height=560):
    g = np.linspace(strength, 0, height).astype(np.uint8)
    a = Image.fromarray(np.repeat(g[:, None], W, 1), 'L')
    sh = Image.new('RGBA', (W, height), (0, 0, 0, 255)); sh.putalpha(a)
    cv.alpha_composite(sh, (0, 0))

def hook(cv, ts):
    k = (ts - T0) / 0.28
    top_shade(cv)
    paste(cv, text('PROMO', ID + 'Inter-Bold.otf', 40, WHITE, track=14), W / 2, 150, alpha=ease_out((ts - T0 - 0.15) / 0.3))
    s = 1.0 + 0.45 * (1 - ease_out(k))
    paste(cv, text('11.11', ANTON, 250, WHITE, track=6), W / 2, 330, scale=s, alpha=min(1, k * 2))
    lw = int(380 * ease_out((ts - T0 - 0.3) / 0.4))
    if lw > 0: ImageDraw.Draw(cv).rectangle((W / 2 - lw / 2, 478, W / 2 + lw / 2, 488), fill=RED + (255,))

def live_badge(cv, ts):
    k = (ts - LIVE_FROM) / 0.3
    if k <= 0: return
    top_shade(cv, 140, 420)
    b = Image.new('RGBA', (340, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(b)
    d.rounded_rectangle((10, 10, 330, 110), 50, fill=RED + (255,))
    pulse = 0.55 + 0.45 * abs(math.sin(ts * 4.0))
    d.ellipse((50, 42, 86, 78), fill=(255, 255, 255, int(255 * pulse)))
    b.alpha_composite(text('LIVE', ARCH, 54, WHITE, shadow=0, pad=2), (108, 30))
    paste(cv, b, W / 2, 230, scale=0.7 + 0.3 * ease_back(k), alpha=min(1, k * 2))

def bg_frame(arr, i):
    return grain(arr.copy(), i, 0.9)

def r_prod(p, ts, s, e, i):
    cv = Image.fromarray(bg_frame(BG_DARK, i).astype(np.uint8)).convert('RGBA')
    k = (ts - s) / 0.32
    kb = 1.0 + 0.05 * (ts - s) / (e - s)
    paste(cv, PROD[p['img']], W / 2, 830 + 40 * (1 - ease_out(k)), scale=(0.94 + 0.06 * ease_out(k)) * kb, alpha=ease_out(k * 1.5))
    paste(cv, text(p['label'], ID + 'Inter-SemiBold.otf', 34, RED, track=10, shadow=0), W / 2, 330, alpha=ease_out(k))
    paste(cv, text(p['idx'] + ' / 03', ID + 'Inter-SemiBold.otf', 30, GREY, track=4, shadow=0), W - 150, 150, alpha=ease_out(k))
    kw = (ts - p['wt']) / 0.22
    if kw > 0:
        paste(cv, text(p['word'], ANTON, 190, WHITE, track=4), W / 2, 1420 + 40 * (1 - ease_out(kw)), alpha=ease_out(kw))
        lw = int(220 * ease_out((ts - p['wt'] - 0.1) / 0.3))
        if lw > 0: ImageDraw.Draw(cv).rectangle((W / 2 - lw / 2, 1560, W / 2 + lw / 2, 1570), fill=RED + (255,))
    return cv

def big_reveal(bg, p, ts, i, lead_c, big_c, tail_c, streak):
    arr = bg_frame(bg, i)
    if streak:  # soft diagonal light sweep
        yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
        pos = (ts - p['lt']) * 900 - 400
        band = np.exp(-(((xx + yy * 0.5) - pos) ** 2) / (2 * 120 ** 2)) * 38
        arr = np.clip(arr + cv2.resize(band, (W, H))[..., None], 0, 255)
    cv = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    k = (ts - p['lt']) / 0.25
    paste(cv, text(p['lead'], ID + 'Inter-Black.otf', 76, lead_c, track=8, shadow=0.3 if streak else 0),
          W / 2, 640 + 30 * (1 - ease_out(k)), alpha=ease_out(k))
    kb = (ts - p['bt']) / 0.22
    if kb > 0:
        shake = 14 * math.exp(-(ts - p['bt']) * 14) * math.sin((ts - p['bt']) * 90)
        paste(cv, text(p['big'], ANTON, 420, big_c, track=8, shadow=0.45 if streak else 0.15),
              W / 2 + shake, 960, scale=1 + 0.5 * (1 - ease_out(kb)), alpha=min(1, kb * 2))
    else:  # anticipation: pulsing dots
        for j in range(3):
            a = 0.35 + 0.65 * max(0, math.sin(ts * 9 - j))
            ImageDraw.Draw(cv).ellipse((W / 2 - 70 + j * 60 - 12, 948, W / 2 - 70 + j * 60 + 12, 972), fill=lead_c + (int(255 * a),))
    kt = (ts - p['tt']) / 0.25
    if kt > 0 and p['tail']:
        paste(cv, text(p['tail'], ANTON if p['tail'].endswith('!') else ID + 'Inter-Black.otf',
                       150 if p['tail'].endswith('!') else 64, tail_c, track=6, shadow=0.3 if streak else 0),
              W / 2, 1300 + 30 * (1 - ease_out(kt)), scale=0.8 + 0.2 * ease_back(kt), alpha=ease_out(kt))
    return cv

def tick_icon(size=110):
    im = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((0, 0, size - 1, size - 1), fill=RED + (255,))
    d.line([(size * .27, size * .52), (size * .44, size * .68), (size * .74, size * .34)], fill=WHITE + (255,), width=int(size * .1), joint='curve')
    return im
TICK = tick_icon()

def stamp_img(txt):
    t = text(txt, ANTON, 120, RED, track=4, shadow=0, pad=6)
    w, h = t.width + 70, t.height + 40
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((4, 4, w - 5, h - 5), 18, fill=(255, 255, 255, 235), outline=RED + (255,), width=8)
    im.alpha_composite(t, (35, 20))
    return im

def r_list(p, ts, s, e, i, base):
    small = cv2.resize(base, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    arr = grain(blur * 0.32 + BG_DARK * 0.55, i, 0.9)
    cv = Image.fromarray(arr.astype(np.uint8)).convert('RGBA')
    k = (ts - s) / 0.25
    paste(cv, text('ADA', ID + 'Inter-Black.otf', 70, RED, track=12, shadow=0), W / 2, 600, alpha=ease_out(k))
    for n, (at, word) in enumerate(p['items']):
        kk = (ts - at) / 0.28
        if kk <= 0: continue
        y = 840 + n * 210
        x_off = -120 * (1 - ease_out(kk))
        paste(cv, TICK, 210 + x_off, y, scale=0.6 + 0.4 * ease_back(kk), alpha=min(1, kk * 2))
        wi = text(word, ANTON, 150, WHITE, track=4)
        paste(cv, wi, 300 + wi.width / 2 - 40 + x_off, y, alpha=ease_out(kk))
    ks = (ts - p['st']) / 0.2
    if ks > 0:
        paste(cv, stamp_img(p['stamp']), W / 2, 1530, scale=1 + 0.6 * (1 - ease_out(ks)), alpha=min(1, ks * 2), rot=-6)
    return cv

def r_end(ts, s, e, i):
    cv = Image.fromarray(bg_frame(BG_DARK, i).astype(np.uint8)).convert('RGBA')
    k = (ts - s) / 0.25
    paste(cv, text('JUMPA', ID + 'Inter-Black.otf', 64, WHITE, track=14, shadow=0), W / 2, 700, alpha=ease_out(k))
    paste(cv, text('11.11', ANTON, 360, RED, track=8, shadow=0), W / 2, 930, scale=1 + 0.3 * (1 - ease_out(k)), alpha=min(1, k * 2))
    lw = int(520 * ease_out((ts - s - 0.15) / 0.4))
    if lw > 0: ImageDraw.Draw(cv).rectangle((W / 2 - lw / 2, 1130, W / 2 + lw / 2, 1136), fill=(255, 255, 255, 200))
    paste(cv, text('AUDIO KERETA  ·  LIVE', ID + 'Inter-SemiBold.otf', 36, GREY, track=10, shadow=0),
          W / 2, 1200, alpha=ease_out((ts - s - 0.25) / 0.3))
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

def frame(base, ts, i):
    for s, e, kind, p in B:
        if s <= ts < e:
            if kind == 'prod': cv = r_prod(p, ts, s, e, i)
            elif kind == 'sale': cv = big_reveal(BG_RED, p, ts, i, WHITE, WHITE, WHITE, True)
            elif kind == 'paper': cv = big_reveal(BG_PAPER, p, ts, i, INK, RED, INK, False)
            elif kind == 'list': cv = r_list(p, ts, s, e, i, base)
            else: cv = r_end(ts, s, e, i)
            break
    else:
        shot = next((a for a in A if a[0] <= ts < a[1]), A[-1])
        cv = Image.fromarray(a_roll(base, shot[2])).convert('RGBA')
        if ts < HOOK_END: hook(cv, ts)
        if ts >= LIVE_FROM: live_badge(cv, ts)
        captions(cv, shot[3], ts)
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
