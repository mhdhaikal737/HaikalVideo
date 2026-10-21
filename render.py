"""Style B (corporate variant) re-edit of the subwoofer/gain talking-head reel.

A-roll = original footage with 3-layer chest captions (no cutouts, no vignette).
B-roll = supplied photos/clips framed as clean cards on a dark-teal page.

Usage: python3 render.py SRC_DIR IMG_DIR OUT.mp4      (env TEST=t1,t2 -> contact jpg)
"""
import os, subprocess, sys, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, IMG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H, FPS = 1080, 1920, 30
DUR = 73.066

TEAL = (63, 224, 176)
LAV = (185, 166, 255)
WHITE = (255, 255, 255)
SOFT = (205, 228, 222)

FD = '/usr/share/fonts/opentype/inter/'
def font(name, size): return ImageFont.truetype(FD + name, size)

# ---------------------------------------------------------------- edit plan
# B shots: (start, end, source, crop, kb=(z0, z1, fx, fy), label, lead, key, tail)
B = [
    (3.80, 5.70, 'img:2', None, (1.00, 1.06, .2, .4), '01 · MASALAH', 'sat-sat ada,', 'TENGGELAM', 'sat-sat hilang'),
    (6.75, 10.45, 'vid:wa1', None, None, '02 · FUNGSI', 'subwoofer keluarkan', 'FREKUENSI RENDAH', ''),
    (10.45, 11.25, 'img:1', None, (1.04, 1.08, .5, .5), '02 · FUNGSI', 'yang tu', 'BASS', ''),
    (13.42, 16.45, 'img:4', None, (1.00, 1.08, .6, .5), '02 · FUNGSI', 'speaker pintu keluarkan', 'VOKAL', 'bunyi suara dalam lagu'),
    (16.45, 18.72, 'img:3', None, (1.00, 1.06, .5, .5), '02 · FUNGSI', 'frekuensi yang', 'TINGGI', ''),
    (21.38, 23.20, 'img:5', (0, 0, 1280, 600), (1.00, 1.07, .3, .5), '03 · PUNCA', 'subwoofer dibuka', 'OVER SANGAT', ''),
    (26.70, 30.82, 'img:9', None, (1.00, 1.10, .5, .45), '03 · PUNCA', 'sebenarnya masalah', 'SETUP BALANCING', 'audio kereta dah lari'),
    (34.02, 35.80, 'img:8', (0, 380, 810, 1000), (1.00, 1.12, .46, .5), '04 · GAIN', 'terutama bahagian', 'GAIN', ''),
    (36.45, 39.28, 'img:7', None, (1.00, 1.30, .30, .55), '04 · GAIN', 'kawal signal', 'KELUAR MASUK', 'ke subwoofer'),
    (42.02, 44.00, 'img:8', (150, 560, 600, 820), (1.00, 1.06, .5, .5), '04 · GAIN', 'gain maksimum', 'BUKAN BASS PADU', ''),
    (46.20, 48.70, 'img:9', (630, 525, 1280, 853), (1.00, 1.05, .5, .5), '05 · BALANCE', 'ada yang keluarkan', 'BASS', ''),
    (52.86, 56.90, 'img:6', None, (1.00, 1.10, .75, .45), '05 · BALANCE', 'setup bagi', 'SAMA-SAMA BALANCE', 'ada bass, ada vokal yang sedap'),
    (59.60, 63.20, 'img:1', None, (1.00, 1.08, .5, .5), '05 · BALANCE', 'bass padu,', 'VOKAL KELUAR', 'baru sedap'),
    (64.30, 68.55, 'vid:wa2', None, None, 'IKUTI KAKAK', 'jangan lupa', 'FOLLOW & LIKE', 'dan komen sekali'),
]
# A captions: (start, end, lead, key, tail, colour, underline)
CAP = [
    (0.00, 1.92, 'pernah tak buka lagu', 'BASS KUAT?', '', TEAL, True),
    (1.92, 3.80, 'tiba-tiba sound', 'VOKAL', 'hampa...', LAV, False),
    (5.70, 6.75, '', 'OKAY,', 'dengar sini', TEAL, False),
    (11.25, 13.42, 'pula,', 'SPEAKER PINTU', '', TEAL, False),
    (18.72, 21.38, 'problem dia sekarang, bila ada', 'SUBWOOFER', '', TEAL, False),
    (23.20, 25.10, 'speaker pintu semua', 'TENGGELAM', '', LAV, True),
    (25.10, 26.70, 'sebenarnya', 'BUKAN', 'masalah speaker pintu', TEAL, False),
    (30.82, 34.02, 'kena faham cara', 'SETUP SUBWOOFER', 'betul-betul', TEAL, True),
    (35.80, 36.45, '', 'GAIN TU APA?', '', LAV, False),
    (39.28, 42.02, 'jangan pusing gain sampai', 'MAKSIMUM', '', LAV, True),
    (44.00, 46.20, 'dalam kereta, ada yang keluarkan', 'BUNYI', '', TEAL, False),
    (48.70, 51.45, 'bass kuat,', 'BUNYI TENGGELAM', '', LAV, False),
    (51.45, 52.86, 'oh,', 'TAK BALANCE', 'sayang', LAV, True),
    (56.90, 59.60, 'senang cerita, kena', 'SETUP BALANCE', '', TEAL, True),
    (63.20, 64.30, '', 'BARU SEDAP', 'okay?', TEAL, False),
    (68.55, 71.70, 'and', 'KOMEN', 'lah sekali', TEAL, False),
    (71.70, 73.10, 'okay lah,', 'BYE!', '', TEAL, False),
]
HANDLE = (68.55, 73.10, '@athirahmajidd_')
# custom tail delays (seconds from caption start) where the VO arrives later
TAIL_AT = {52.86: 1.80, 26.70: 2.20, 13.42: 1.30}


def shot_at(t):
    for i, b in enumerate(B):
        if b[0] <= t < b[1]:
            return 'B', i
    # A shots are the gaps between B shots
    k = sum(1 for b in B if b[1] <= t)
    return 'A', k


# ---------------------------------------------------------------- text layers
_cache = {}
def text_layer(txt, fnt, color, shadow=True, maxw=960):
    key = (txt, fnt.path, fnt.size, color, maxw)
    if key in _cache: return _cache[key]
    f = fnt
    while f.size > 30 and f.getlength(txt) > maxw:
        f = ImageFont.truetype(f.path, int(f.size * 0.93))
    bb = f.getbbox(txt)
    pad = 24
    w, h = int(bb[2] - bb[0]) + pad * 2, int(bb[3] - bb[1]) + pad * 2
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), txt, font=f, fill=color + (255,))
    if shadow:
        a = im.split()[3].filter(ImageFilter.GaussianBlur(9))
        sh = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        sh.putalpha(a.point(lambda v: int(v * 0.6)))
        sh = Image.alpha_composite(sh, im)
        im = sh
    arr = np.asarray(im).astype(np.float32)
    _cache[key] = (arr, pad)
    return _cache[key]


def paste(frame, arr, x, y, alpha=1.0):
    """frame float32 HxWx3 RGB, arr RGBA float, top-left x,y"""
    h, w = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    a = arr[y0 - y:y1 - y, x0 - x:x1 - x]
    al = a[..., 3:4] / 255.0 * alpha
    frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - al) + a[..., :3] * al


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def draw_caption(frame, t, t0, lead, key, tail, color, ul, ytop, tail_at=None):
    """3-layer centred caption: small lead, big keyword, italic tail."""
    layers = []
    if lead: layers.append((text_layer(lead, font('Inter-SemiBold.otf', 46), WHITE), 0.0, 6))
    kl = text_layer(key, font('Inter-ExtraBold.otf', 104), color)
    layers.append((kl, 0.10, 0))
    if tail: layers.append((text_layer(tail, font('Inter-MediumItalic.otf', 46), SOFT), tail_at if tail_at else 0.22, 0))
    y = ytop
    for (arr, pad), d, gap in layers:
        p = ease((t - t0 - d) / 0.25)
        if p > 0:
            h, w = arr.shape[:2]
            x = (W - w) // 2
            paste(frame, arr, x, int(y + (1 - p) * 22), p)
            if arr is kl[0] and ul:
                q = ease((t - t0 - d - 0.2) / 0.35)
                kw = w - 2 * pad
                lw = int(kw * 0.55 * q)
                if lw > 2:
                    cy = y + h - pad + 24
                    frame[cy:cy + 6, W // 2 - lw // 2:W // 2 + lw // 2] = np.array(color, np.float32)
        y += arr.shape[0] - 2 * pad + 18 + gap + (30 if arr is kl[0] else 0)


# ---------------------------------------------------------------- B-roll page
def make_bg():
    yy = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([14, 38, 34], np.float32), np.array([20, 58, 51], np.float32)
    bg = top * (1 - yy) + bot * yy
    bg = np.repeat(bg, W, axis=1)
    # one soft teal glow behind the card (kept low, no vignette)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    g = np.exp(-(((gx - W / 2) / 620) ** 2 + ((gy - 800) / 620) ** 2))
    bg += g[..., None] * np.array([18, 48, 40], np.float32)
    rng = np.random.default_rng(3)
    bg += rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
    return np.clip(bg, 0, 255)
BG = make_bg()

CARD_W, CARD_MAXH, CARD_TOP = 960, 980, 330

def load_img(n, crop):
    im = cv2.cvtColor(cv2.imread(f'{IMG}/{n}.jpg'), cv2.COLOR_BGR2RGB)
    if crop: im = im[crop[1]:crop[3], crop[0]:crop[2]]
    return im

def card_geom(aspect):
    ch = min(CARD_MAXH, int(CARD_W / aspect))
    return CARD_W, ch

_mask = {}
def round_mask(w, h, r=30):
    if (w, h) not in _mask:
        m = Image.new('L', (w, h), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
        _mask[(w, h)] = np.asarray(m).astype(np.float32)[..., None] / 255
        sh = Image.new('L', (w + 120, h + 120), 0)
        ImageDraw.Draw(sh).rounded_rectangle((60, 75, w + 59, h + 74), r, fill=150)
        _mask[('s', w, h)] = np.asarray(sh.filter(ImageFilter.GaussianBlur(26))).astype(np.float32)[..., None] / 255
        bd = Image.new('L', (w, h), 0)
        ImageDraw.Draw(bd).rounded_rectangle((0, 0, w - 1, h - 1), r, outline=255, width=2)
        _mask[('b', w, h)] = np.asarray(bd).astype(np.float32)[..., None] / 255
    return _mask[(w, h)], _mask[('s', w, h)], _mask[('b', w, h)]


def fit_cover(im, cw, ch, z, fx, fy):
    """crop im to card aspect, zoom z around normalised focus, resize to cw x ch"""
    ih, iw = im.shape[:2]
    a = cw / ch
    if iw / ih > a: bw, bh = ih * a, ih
    else: bw, bh = iw, iw / a
    bw, bh = bw / z, bh / z
    cx = min(max(fx * iw, bw / 2), iw - bw / 2)
    cy = min(max(fy * ih, bh / 2), ih - bh / 2)
    sx, sy = cw / bw, ch / bh
    M = np.float32([[sx, 0, -(cx - bw / 2) * sx], [0, sy, -(cy - bh / 2) * sy]])
    return cv2.warpAffine(im, M, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def read_clip(name):
    p = f'{SRC}/{name}.mp4'
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=width,height',
                          '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip().split(',')
    w, h = int(out[0]), int(out[1])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', f'fps={FPS}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
CLIPS = {'wa1': read_clip('wa1'), 'wa2': read_clip('wa2')}
IMGS = {}

def b_frame(i, t):
    s, e, src, crop, kb, label, lead, key, tail = B[i]
    fr = BG.copy()
    lt = t - s
    kind, name = src.split(':')
    if kind == 'img':
        k = (name, crop)
        if k not in IMGS: IMGS[k] = load_img(name, crop)
        im = IMGS[k]
        cw, ch = card_geom(im.shape[1] / im.shape[0])
        z0, z1, fx, fy = kb
        z = z0 + (z1 - z0) * (lt / (e - s))
        card = fit_cover(im, cw, ch, z, fx, fy)
    else:
        frames = CLIPS[name]
        im = frames[min(int(lt * FPS), len(frames) - 1)]
        cw, ch = card_geom(im.shape[1] / im.shape[0])
        if name == 'wa2':  # phone-sized screen card
            cw = 820; ch = int(cw * im.shape[0] / im.shape[1])
        card = fit_cover(im, cw, ch, 1.0, .5, .42)
    card = card.astype(np.float32)
    m, sm, bd = round_mask(cw, ch)
    p = ease(lt / 0.28)
    x = (W - cw) // 2
    y = CARD_TOP + (CARD_MAXH - ch) // 2 - 40 + int((1 - p) * 26)
    # shadow
    sh_h, sh_w = sm.shape[:2]
    ys, xs = y - 60, x - 60
    reg = fr[ys:ys + sh_h, xs:xs + sh_w]
    reg[:] = reg * (1 - sm * 0.8 * p)
    card = card * (1 - bd * 0.25) + 255 * bd * 0.25
    al = m * p
    fr[y:y + ch, x:x + cw] = fr[y:y + ch, x:x + cw] * (1 - al) + card * al
    # running label: letter-spaced caps + hairline
    lab = text_layer(' '.join(label), font('Inter-SemiBold.otf', 34), (143, 217, 196), shadow=False)
    paste(fr, lab[0], 60 - lab[1], 190 - lab[1], 1.0)
    fr[248:250, 60:W - 60] = fr[248:250, 60:W - 60] * 0.6 + np.array([143, 217, 196], np.float32) * 0.4
    cap_top = y + ch + 50
    draw_caption(fr, t, s, lead, key, tail, TEAL, False, cap_top, TAIL_AT.get(s))
    return fr


# ---------------------------------------------------------------- A-roll
def a_frame(src, k, t):
    z = 1.0 if k % 2 == 0 else 1.08
    fx, fy = 540, 720
    if z != 1.0:
        M = np.float32([[z, 0, fx - z * fx], [0, z, fy - z * fy]])
        src = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR)
    fr = src.astype(np.float32)
    # light corporate grade: a touch of contrast, slightly cooler, no vignette
    fr = (fr - 128) * 1.06 + 128
    lum = fr.mean(axis=2, keepdims=True)
    fr = lum + (fr - lum) * 0.92
    fr[..., 2] += 3
    # soft bottom gradient so captions read cleanly
    fr[1050:] *= GRAD
    for c in CAP:
        if c[0] <= t < c[1]:
            draw_caption(fr, t, c[0], c[2], c[3], c[4], c[5], c[6], 1240)
    if t < 3.8:  # hook tag
        p = ease(t / 0.3)
        tag = text_layer('T I P S   A U D I O   K E R E T A', font('Inter-SemiBold.otf', 32), TEAL, shadow=False)
        tw, th = tag[0].shape[1] - 2 * tag[1], tag[0].shape[0] - 2 * tag[1]
        pw, ph = tw + 72, th + 40
        pill = np.zeros((ph, pw, 4), np.float32)
        pm = Image.new('L', (pw, ph), 0)
        ImageDraw.Draw(pm).rounded_rectangle((0, 0, pw - 1, ph - 1), ph // 2, fill=225)
        pill[..., :3] = np.array([14, 38, 34], np.float32)
        pill[..., 3] = np.asarray(pm)
        x0 = (W - pw) // 2
        paste(fr, pill, x0, 200, p)
        paste(fr, tag[0], x0 + 36 - tag[1], 220 - tag[1], p)
    if t >= HANDLE[0]:
        p = ease((t - HANDLE[0] - 0.3) / 0.3)
        hl = text_layer(HANDLE[2], font('Inter-SemiBold.otf', 40), WHITE)
        hw, hh = hl[0].shape[1] - 2 * hl[1], hl[0].shape[0] - 2 * hl[1]
        x0, y0 = (W - hw) // 2 - 34, 1600
        pill = np.zeros((hh + 36, hw + 68, 4), np.float32)
        pm = Image.new('L', (hw + 68, hh + 36), 0)
        ImageDraw.Draw(pm).rounded_rectangle((0, 0, hw + 67, hh + 35), (hh + 36) // 2, fill=200)
        pill[..., :3] = np.array([20, 58, 51], np.float32)
        pill[..., 3] = np.asarray(pm)
        paste(fr, pill, x0, y0, p)
        paste(fr, hl[0], x0 + 34 - hl[1], y0 + 18 - hl[1], p)
    return fr

GRAD = (1 - 0.38 * np.clip((np.arange(1050, H) - 1050) / 870, 0, 1) ** 1.4)[:, None, None].astype(np.float32)


# ---------------------------------------------------------------- main loop
def frame_at(src, t):
    kind, k = shot_at(t)
    fr = b_frame(k, t) if kind == 'B' else a_frame(src, k, t)
    return np.clip(fr, 0, 255).astype(np.uint8)


def decoder(start=0.0):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', f'{SRC}/main.mov', '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)

if os.environ.get('TEST'):
    ts = [float(x) for x in os.environ['TEST'].split(',')]
    tiles = []
    for t in ts:
        p = decoder(t)
        src = np.frombuffer(p.stdout.read(W * H * 3), np.uint8).reshape(H, W, 3)
        p.kill()
        tiles.append(cv2.resize(frame_at(src, t), (int(os.environ.get('TW', 360)), int(os.environ.get('TW', 360)) * 16 // 9)))
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(OUT, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    sys.exit()

dec = decoder()
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                        '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', OUT],
                       stdin=subprocess.PIPE)
n = 0
while True:
    buf = dec.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    src = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    enc.stdin.write(frame_at(src, n / FPS).tobytes())
    n += 1
    if n % 300 == 0: print(n, flush=True)
enc.stdin.close(); enc.wait()
print('frames', n)
