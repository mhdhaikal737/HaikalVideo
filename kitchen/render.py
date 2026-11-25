"""Re-edit of the under-seat subwoofer talking-head reel in the style of the
reference reel: mostly original footage, short white lead-ins + bold white
keywords on a royal-blue box, real-photo B-roll (CC licensed) and plain
light-grey cards with flat blue icons. The burned-in hook (0-5.5 s) is left
untouched.

Usage: python3 render.py SRC BROLL_DIR OUT.mp4     (env TEST=t1,t2 -> contact jpg)
"""
import os, subprocess, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, BR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H, FPS = 1080, 1920, 30
DUR = 57.07
HOOK_END = 5.50                      # original hook text is visible until here

BLUE = (34, 52, 168)
NAVY = (22, 34, 110)
WHITE = (255, 255, 255)
GREY = (90, 96, 110)
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts/')
def font(w, size): return ImageFont.truetype(FD + f'Montserrat-{w}.ttf', size)

# ------------------------------------------------------------------ edit plan
# A-roll captions: (start, end, lead, [(word, boxed)], y_centre)
CAP = [
    (6.60, 8.40, 'contoh depan', [('INI', True)], 330),
    (8.46, 10.40, 'dua-dua', [('10 INCI', True)], 330),
    (10.44, 12.60, 'speaker dalam ni', [('SAMA', True)], 330),
    (12.70, 13.55, 'hat ni pun', [('SAMA', False)], 330),
    (13.58, 15.88, 'yang beza dia', [('CASING', True)], 330),
    (15.90, 17.50, 'dia punya', [('BENTUK', True)], 330),
    (17.54, 18.80, 'hat ni', [('PANJANG', True)], 330),
    (18.82, 19.42, 'hat ni', [('PENDEK', True)], 330),
    (22.54, 24.54, 'lagi tebal daripada', [('10 INCI', True)], 330),
    (24.56, 26.00, 'lepas tu', [('10 INCI', True)], 330),
    (26.04, 27.48, '', [('SLIM,', False), ('KECIL', True)], 330),
    (27.50, 28.56, 'macam', [('8 INCI', True)], 330),
    (28.58, 29.78, 'sebelum hampa', [('BELI', True)], 330),
    (35.66, 36.50, 'yang', [('KEDUA', True)], 330),
    (38.46, 39.26, 'yang', [('KETIGA', True)], 330),
    (42.82, 44.18, 'so hampa', [('DAPAT', False), ('KAN?', True)], 330),
    (45.54, 47.88, '', [('NEXT', False), ('VIDEO', True)], 330),
    (49.54, 50.90, 'nak tahu', [('TAK?', True)], 330),
    (50.90, 52.28, 'jangan lupa', [('FOLLOW', True)], 330),
    (52.28, 53.74, 'kakak punya video', [('SHARE', True)], 330),
    (53.74, 55.02, 'jangan lupa', [('LIKE,', False), ('KOMEN', True)], 330),
    (55.04, 57.07, 'jumpa kakak in the', [('NEXT VIDEO', True)], 330),
]

# B-roll: (start, end, kind, arg, lead, [(word, boxed)])
#   kind 'img': arg = (file, crop_x0, crop_y0, crop_h, zoom_from, zoom_to)
#   kind 'card': arg = card name
B = [
    (5.50, 6.58, 'img', ('c15.jpg', 690, 0, 560, 1.00, 1.06), 'sangkut dekat', [('SEAT BESI', True)]),
    (19.44, 20.42, 'card', 'shape', '', []),
    (20.44, 22.52, 'img', ('c12.jpg', 282, 0, 819, 1.08, 1.00), 'ada je subwoofer', [('8 INCI', True)]),
    (29.80, 31.68, 'img', ('c16.jpg', 110, 0, 768, 1.00, 1.08), 'check dulu', [('RUANG BAWAH SEAT', True)]),
    (31.70, 33.80, 'card', 'three', '', []),
    (33.82, 35.64, 'card', 'list1', '', []),
    (36.52, 38.44, 'img', ('c06.jpg', 0, 0, 768, 1.00, 1.07), 'lebar antara', [('2 TRACK SEAT', True)]),
    (39.28, 40.98, 'card', 'list3', '', []),
    (41.00, 42.80, 'img', ('c09.jpg', 120, 0, 576, 1.00, 1.08), '', [('DAH UKUR,', True)]),
    (47.90, 49.52, 'card', 'rms', '', []),
]

# ------------------------------------------------------------------ helpers
def ease_back(x):
    x = min(max(x, 0.0), 1.0); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

def ease_out(x):
    x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3

def shadowed(img, blur=10, alpha=110, off=(0, 6)):
    """RGBA image -> same image with a soft drop shadow (padded)."""
    pad = blur * 3
    out = Image.new('RGBA', (img.width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    a = img.split()[3].point(lambda v: v * alpha // 255)
    sh = Image.new('RGBA', img.size, (0, 0, 0, 0)); sh.putalpha(a)
    out.alpha_composite(sh, (pad + off[0], pad + off[1]))
    out = out.filter(ImageFilter.GaussianBlur(blur))
    out.alpha_composite(img, (pad, pad))
    return out

def text_img(txt, fnt, fill):
    l, t, r, b = fnt.getbbox(txt)
    im = Image.new('RGBA', (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t), txt, font=fnt, fill=fill)
    return im

def keyword_row(words, size=92):
    """Row of keywords; boxed words get a royal-blue rounded box."""
    f = font(900, size)
    parts = []
    for w, boxed in words:
        t = text_img(w, f, WHITE)
        if boxed:
            px, py = 22, 14
            bx = Image.new('RGBA', (t.width + 2 * px, t.height + 2 * py), (0, 0, 0, 0))
            ImageDraw.Draw(bx).rounded_rectangle((0, 0, bx.width - 1, bx.height - 1), 10, fill=BLUE + (255,))
            bx.alpha_composite(t, (px, py))
            parts.append(bx)
        else:
            px, py = 6, 14
            bx = Image.new('RGBA', (t.width + 2 * px, t.height + 2 * py), (0, 0, 0, 0))
            bx.alpha_composite(t, (px, py)); parts.append(bx)
    gap = 14
    wsum = sum(p.width for p in parts) + gap * (len(parts) - 1)
    hmax = max(p.height for p in parts)
    row = Image.new('RGBA', (wsum, hmax), (0, 0, 0, 0)); x = 0
    for p in parts:
        row.alpha_composite(p, (x, (hmax - p.height) // 2)); x += p.width + gap
    if row.width > W - 140:                      # keep inside safe margin
        s = (W - 140) / row.width
        row = row.resize((int(row.width * s), int(row.height * s)), Image.LANCZOS)
    return shadowed(row, blur=8, alpha=90, off=(0, 5))

def lead_img(txt, size=46, fill=WHITE, shadow=True):
    t = text_img(txt, font(600, size), fill)
    return shadowed(shadowed(t, blur=3, alpha=230, off=(0, 2)), blur=9, alpha=120, off=(0, 0)) if shadow else t

def paste_scaled(canvas, im, cx, cy, scale=1.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0.01: return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 0.999:
        im = im.copy(); im.putalpha(im.split()[3].point(lambda v: int(v * alpha)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

# ------------------------------------------------------------------ caption layers
CAP_L = [(lead_img(l) if l else None, keyword_row(k)) for _, _, l, k, _ in CAP]
BCAP_L = [((lead_img(l) if l else None), (keyword_row(k) if k else None)) for *_, l, k in B]

def draw_caption(canvas, lead, key, t, cy):
    if lead is not None:
        a = ease_out(t / 0.18)
        paste_scaled(canvas, lead, W / 2, cy - key.height / 2 - 8 + (1 - a) * 14, 1.0, a)
    k = ease_back((t - 0.06) / 0.22)
    paste_scaled(canvas, key, W / 2, cy + 22, 0.82 + 0.18 * k, min(1.0, max(0.0, (t - 0.06) / 0.1)))

# ------------------------------------------------------------------ B-roll images
IMG_CACHE = {}
def broll_img(arg, p):
    fn, x0, y0, ch, z0, z1 = arg
    if fn not in IMG_CACHE:
        im = Image.open(os.path.join(BR, fn)).convert('RGB')
        cw = int(ch * W / H)
        crop = im.crop((x0, y0, x0 + cw, y0 + ch))
        big = crop.resize((int(W * 1.1), int(H * 1.1)), Image.LANCZOS).filter(
            ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
        IMG_CACHE[fn] = np.asarray(big)
    big = IMG_CACHE[fn]
    z = (z0 + (z1 - z0) * p) / 1.1
    import cv2
    bh, bw = big.shape[:2]
    cw, chh = bw * z, bh * z
    x, y = (bw - cw) / 2, (bh - chh) / 2
    M = np.array([[cw / W, 0, x], [0, chh / H, y]], np.float32)
    return cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

# ------------------------------------------------------------------ cards (light grey, flat blue)
def card_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.42) / H) ** 2)
    v = np.clip(1 - d * 0.55, 0, 1)
    top = np.array([247, 248, 251], np.float32); edge = np.array([214, 218, 228], np.float32)
    img = edge + (top - edge) * v[..., None]
    rng = np.random.default_rng(3)
    img += rng.normal(0, 1.6, (H, W, 1))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGBA')
BG = card_bg()

SS = 4  # supersample for icons
def icon(kind, size, col=BLUE):
    S = size * SS
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    lw = int(S * 0.085)
    if kind == 'check':
        d.ellipse((0, 0, S - 1, S - 1), fill=col + (255,))
        d.line([(S * .28, S * .52), (S * .44, S * .67), (S * .73, S * .36)], fill=WHITE, width=int(S * .1), joint='curve')
    elif kind == 'height':
        d.line([(S * .5, S * .14), (S * .5, S * .86)], fill=col, width=lw)
        d.polygon([(S * .5, S * .02), (S * .32, S * .24), (S * .68, S * .24)], fill=col)
        d.polygon([(S * .5, S * .98), (S * .32, S * .76), (S * .68, S * .76)], fill=col)
        d.line([(S * .18, S * .02 + lw / 2), (S * .82, S * .02 + lw / 2)], fill=col, width=lw // 2)
    elif kind == 'width':
        d.rectangle((S * .02, S * .15, S * .14, S * .85), fill=col)
        d.rectangle((S * .86, S * .15, S * .98, S * .85), fill=col)
        d.line([(S * .26, S * .5), (S * .74, S * .5)], fill=col, width=lw)
        d.polygon([(S * .18, S * .5), (S * .36, S * .34), (S * .36, S * .66)], fill=col)
        d.polygon([(S * .82, S * .5), (S * .64, S * .34), (S * .64, S * .66)], fill=col)
    elif kind == 'hole':
        d.ellipse((S * .08, S * .08, S * .92, S * .92), outline=col, width=lw)
        d.ellipse((S * .32, S * .32, S * .68, S * .68), fill=col)
    elif kind == 'wave':
        pts = [(S * (0.04 + 0.92 * i / 60), S * (0.5 - 0.36 * math.sin(i / 60 * math.pi * 3) * (0.4 + 0.6 * i / 60)))
               for i in range(61)]
        d.line(pts, fill=col, width=lw, joint='curve')
    return im.resize((size, size), Image.LANCZOS)

def big_blue(txt, size):
    t = text_img(txt, font(900, size), BLUE)
    return shadowed(t, blur=6, alpha=60, off=(0, 4))

def small_grey(txt, size=40):
    return text_img(txt, font(600, size), GREY)

def list_card(num, ic, txt, w=880, h=190):
    c = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(c)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 26, fill=(255, 255, 255, 255))
    d.rounded_rectangle((0, 0, 16, h - 1), 8, fill=BLUE + (255,))
    d.rectangle((8, 0, 16, h - 1), fill=(255, 255, 255, 255))
    d.rounded_rectangle((0, 0, 12, h - 1), 6, fill=BLUE + (255,))
    c.alpha_composite(icon(ic, 96), (60, (h - 96) // 2))
    f = font(900, 50)
    lines = txt.split('\n'); lh = 58; y0 = (h - lh * len(lines)) // 2
    for i, ln in enumerate(lines):
        d.text((200, y0 + i * lh), ln, font=f, fill=NAVY)
    d.text((w - 70, 22), num, font=font(800, 30), fill=(150, 165, 230))
    return shadowed(c, blur=14, alpha=55, off=(0, 8))

LIST = [list_card('01', 'height', 'TINGGI'),
        list_card('02', 'width', 'LEBAR ANTARA\n2 TRACK SEAT'),
        list_card('03', 'hole', 'TAK TUTUP\nLUBANG')]

def shape_icon(long_box, size=(380, 260)):
    """Flat outline of an under-seat sub box: long & low vs short & tall."""
    w, h = size; S = SS
    im = Image.new('RGBA', (w * S, h * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    bw, bh = (w * .96, h * .48) if long_box else (w * .66, h * .78)
    x0 = (w - bw) / 2; y0 = h - bh - 4
    d.rounded_rectangle((x0 * S, y0 * S, (x0 + bw) * S, (y0 + bh) * S), 22 * S, outline=BLUE, width=9 * S)
    r = min(bw, bh) * .36; cx, cy = w / 2, y0 + bh / 2
    d.ellipse(((cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S), outline=BLUE, width=9 * S)
    d.ellipse(((cx - r * .35) * S, (cy - r * .35) * S, (cx + r * .35) * S, (cy + r * .35) * S), fill=BLUE)
    return im.resize((w, h), Image.LANCZOS)

CARD_EL = {
    'shape': [shape_icon(True), shape_icon(False), small_grey('tak sama', 44), big_blue('BENTUK', 120)],
    'three': [icon('check', 150), small_grey('ada', 44), big_blue('3 BENDA', 128), big_blue('WAJIB CHECK', 92)],
    'rms': [small_grey('next video, nak tahu pasal', 40), icon('wave', 130), big_blue('RMS', 190)],
    'list_head': small_grey('yang hampa wajib check', 40),
}

def draw_card(name, t):
    c = BG.copy()
    pop = lambda d: ease_back((t - d) / 0.24)
    fade = lambda d: min(1.0, max(0.0, (t - d) / 0.12))
    if name == 'shape':
        a, b, s, k = CARD_EL['shape']
        paste_scaled(c, a, W / 2, 640, 0.85 + 0.15 * pop(0.0), fade(0.0))
        paste_scaled(c, b, W / 2, 960, 0.85 + 0.15 * pop(0.14), fade(0.14))
        paste_scaled(c, s, W / 2, 1180, 1, fade(0.2))
        paste_scaled(c, k, W / 2, 1290, 0.85 + 0.15 * pop(0.26), fade(0.26))
    elif name == 'three':
        ic, s, k1, k2 = CARD_EL['three']
        paste_scaled(c, ic, W / 2, 700, pop(0.0), fade(0.0))
        paste_scaled(c, s, W / 2, 860, 1, fade(0.1))
        paste_scaled(c, k1, W / 2, 970, 0.85 + 0.15 * pop(0.3), fade(0.3))
        paste_scaled(c, k2, W / 2, 1090, 0.85 + 0.15 * pop(1.5), fade(1.5))
    elif name in ('list1', 'list3'):
        paste_scaled(c, CARD_EL['list_head'], W / 2, 560, 1, 1 if name == 'list3' else fade(0))
        for i, lc in enumerate(LIST):
            if name == 'list1' and i > 0: break
            d = 0.0 if name == 'list1' else (0.0 if i < 2 else 0.35)
            if name == 'list3' and i < 2:
                paste_scaled(c, lc, W / 2, 760 + i * 230, 1, 1)
            else:
                paste_scaled(c, lc, W / 2, 760 + i * 230, 0.9 + 0.1 * pop(d), fade(d))
    elif name == 'rms':
        s, ic, k = CARD_EL['rms']
        paste_scaled(c, s, W / 2, 720, 1, fade(0))
        paste_scaled(c, ic, W / 2, 860, 1, fade(0.05))
        paste_scaled(c, k, W / 2, 1030, 0.8 + 0.2 * pop(0.12), fade(0.12))
    return c

# ------------------------------------------------------------------ frame composer
def compose(t, frame):
    """frame: HxWx3 uint8 RGB of the original at time t."""
    for i, (s, e, kind, arg, *_rest) in enumerate(B):
        if s <= t < e:
            lt = t - s
            if kind == 'img':
                base = Image.fromarray(broll_img(arg, lt / (e - s))).convert('RGBA')
                lead, key = BCAP_L[i]
                if key is not None:
                    draw_caption(base, lead, key, lt, 1320)
                return np.asarray(base.convert('RGB'))
            return np.asarray(draw_card(arg, lt).convert('RGB'))
    if t < HOOK_END:
        return frame                         # hook untouched
    for i, (s, e, *_r) in enumerate(CAP):
        if s <= t < e:
            base = Image.fromarray(frame).convert('RGBA')
            lead, key = CAP_L[i]
            draw_caption(base, lead, key, t - s, CAP[i][4])
            return np.asarray(base.convert('RGB'))
    return frame

def decode(t0=0, dur=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(t0), '-i', SRC]
    if dur: cmd += ['-t', str(dur)]
    cmd += ['-vf', f'scale={W}:{H},fps={FPS}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)

if os.environ.get('TEST'):
    ts = [float(x) for x in os.environ['TEST'].split(',')]
    tiles = []
    for t in ts:
        p = decode(t, 0.03); fr = np.frombuffer(p.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3)
        p.stdout.close(); p.kill(); p.wait()
        im = Image.fromarray(compose(t, fr)).resize((360, 640))
        ImageDraw.Draw(im).text((8, 8), f'{t:.2f}', font=font(800, 28), fill=(255, 40, 40))
        tiles.append(im)
    cols = min(6, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * 360, rows * 640), 'black')
    for i, im in enumerate(tiles): sheet.paste(im, ((i % cols) * 360, (i // cols) * 640))
    sheet.save(OUT); sys.exit()

dec = decode()
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                        '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], stdin=subprocess.PIPE)
n = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    enc.stdin.write(np.ascontiguousarray(compose(n / FPS, fr)).tobytes())
    n += 1
    if n % 150 == 0: print('frame', n, flush=True)
enc.stdin.close(); enc.wait(); dec.wait()
print('done', n)
