"""Synth music bed (+ optional SFX track, not used in the v3 mix) for render.py's edit plan (times in source seconds, shifted by T0).

Usage: python3 audio.py MUSIC.wav SFX.wav
"""
import sys, wave
import numpy as np

SR = 44100
T0, T1 = 0.60, 28.90
DUR = T1 - T0
N = int(SR * DUR)
rng = np.random.default_rng(5)

CUTS = [1.98, 2.90, 3.95, 4.68, 5.30, 7.32, 9.30, 11.80, 15.24, 16.36, 18.48, 19.70, 21.74, 23.10, 25.24, 26.40, 28.10]
BOOMS = [0.66, 8.04, 22.08, 28.12]
CONFETTI = [0.66, 8.04, 13.88, 22.08, 28.12]
POPS = [1.98, 3.68, 4.40, 4.78, 6.04, 10.42, 12.22, 13.16, 14.46, 15.78, 17.90, 19.30, 20.60, 22.72, 24.00, 27.16]
STAMP = 13.88
LIVE = 25.54
DROP = (7.00, 8.04)   # music drops out under the riser, back on the 11.11 slam

def env_exp(n, k): return np.exp(-np.arange(n) / SR * k)

def add(buf, x, t, g=1.0):
    s = int((t - T0) * SR)
    if s < 0: x, s = x[-s:], 0
    e = min(len(buf), s + len(x))
    if e > s: buf[s:e] += x[:e - s] * g

def lowpass(x, c):
    y = np.zeros_like(x); a = 0.0
    for i in range(len(x)): a += c * (x[i] - a); y[i] = a
    return y

# ---------------------------------------------------------------- sfx
def whoosh(d=0.34):
    n = int(SR * d); x = rng.normal(0, 1, n)
    y = np.zeros(n); a = 0.0
    for i in range(n):
        c = 0.02 + 0.2 * np.sin(np.pi * i / n); a += c * (x[i] - a); y[i] = a
    return y * np.sin(np.pi * np.arange(n) / n) ** 2 / np.abs(y).max()

def boom(d=1.4):
    n = int(SR * d); t = np.arange(n) / SR
    f = 42 + 80 * np.exp(-t * 9)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 3.2)
    click = rng.normal(0, 1, n) * env_exp(n, 60) * 0.35
    return np.tanh((body + click) * 1.6)

def pop(d=0.09):
    n = int(SR * d); t = np.arange(n) / SR
    f = 900 * np.exp(-t * 18) + 380
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 45)

def riser(d):
    n = int(SR * d); t = np.arange(n) / SR
    f = 180 * 2 ** (t / d * 3)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5
    noise = lowpass(rng.normal(0, 1, n), 0.25) * 1.2
    return (tone + noise) * (t / d) ** 2.2

def stamp(d=0.35):
    n = int(SR * d)
    return np.tanh((lowpass(rng.normal(0, 1, n), 0.18) * 3 + boom(d) * 0.8) * env_exp(n, 14))

def ding(d=0.9):
    n = int(SR * d); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1318 * t) + 0.6 * np.sin(2 * np.pi * 1976 * t)) * env_exp(n, 6) * 0.5

sfx = np.zeros(N)
w = whoosh()
for c in CUTS: add(sfx, w, c - 0.2, 0.22)
for b in BOOMS: add(sfx, boom(), b, 0.55)
for p in POPS: add(sfx, pop(), p, 0.16)
add(sfx, riser(DROP[1] - 7.32), 7.32, 0.22)
add(sfx, stamp(), STAMP, 0.5)
add(sfx, ding(), LIVE, 0.22)
for c in CONFETTI:
    for j, f in enumerate((2093, 2637, 3136)):
        add(sfx, np.sin(2 * np.pi * f * np.arange(int(SR * .5)) / SR) * env_exp(int(SR * .5), 9), c + 0.05 + j * 0.06, 0.07)

# ---------------------------------------------------------------- music bed (A minor, 120 bpm)
BEAT = 0.5
mus = np.zeros(N)
def note(m): return 440 * 2 ** ((m - 69) / 12)
PROG = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]  # Am F C G

def kick():
    n = int(SR * 0.32); t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 11)

def hat():
    n = int(SR * 0.05); x = rng.normal(0, 1, n)
    return (x - lowpass(x, 0.5)) * env_exp(n, 90)

def saw(f, d):
    n = int(SR * d); t = np.arange(n) / SR
    return 2 * ((t * f) % 1) - 1

K, Hh = kick(), hat()
t = T0
bar = 0
while t < T1 - 0.3:
    root, chord = PROG[(bar // 2) % 4]
    for b in range(4):
        tb = t + b * BEAT
        add(mus, K, tb, 0.55)
        add(mus, Hh, tb + BEAT / 2, 0.18)
        for h in range(2):  # 8th-note sub bass
            n = int(SR * BEAT / 2)
            bs = np.sin(2 * np.pi * note(root - 12) * np.arange(n) / SR)
            bs = bs * np.minimum(1, np.arange(n) / 300) * np.minimum(1, (n - np.arange(n)) / 1200)
            add(mus, bs, tb + h * BEAT / 2, 0.30)
    pad = sum(saw(note(m) * dt, BEAT * 4) for m in chord for dt in (0.997, 1.003))
    pad = lowpass(pad, 0.035)
    n = len(pad); fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / 2000)
    add(mus, pad * fade, t, 0.045)
    t += BEAT * 4; bar += 1

# drop out under the riser, fade out at the end card
tt = np.arange(N) / SR + T0
g = np.ones(N)
g = lowpass(g, 0.002)
g *= np.clip((T1 - tt) / 0.8, 0, 1)
mus *= g

def write(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())

write(sys.argv[1], mus / max(1e-9, np.abs(mus).max()) * 0.5)
write(sys.argv[2], sfx)
