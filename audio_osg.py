"""SFX bed for render_osg.py: whoosh on B-roll cuts, pop on tags / punch words,
a tick on the price strike. Nothing before 3.4 s so the hook audio stays as-is.

Usage: python3 audio_osg.py OUT_SFX.wav
"""
import sys, wave
import numpy as np

SR = 44100
DUR = 54.5
# B-roll in/out cuts from render_osg.py PLAN
CUTS = [5.96, 8.58, 9.35, 11.96, 15.45, 18.62, 36.53, 37.68, 40.16, 41.94, 42.98, 46.70, 47.62, 49.30, 50.87, 52.95]
# tag pills, punch words and emoji pops
POPS = [7.15, 8.05, 8.58, 13.35, 15.0, 16.10, 17.20, 17.82, 18.95, 20.50, 22.3, 35.50, 36.75, 37.25, 39.72,
        41.20, 42.5, 45.55, 46.74, 48.55, 49.95, 53.4]
TICKS = [46.25]

rng = np.random.default_rng(7)

def whoosh(d=0.34):
    n = int(SR * d)
    x = rng.normal(0, 1, n)
    y = np.zeros(n); a = 0.0
    for i in range(n):
        c = 0.03 + 0.17 * np.sin(np.pi * i / n)
        a += c * (x[i] - a); y[i] = a
    env = np.sin(np.pi * np.arange(n) / n) ** 2
    return y * env / np.abs(y).max()

def pop(d=0.09):
    t = np.arange(int(SR * d)) / SR
    f = 950 - 530 * t / d
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45)

def tick(d=0.05):
    t = np.arange(int(SR * d)) / SR
    return np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 90)

out = np.zeros(int(SR * DUR))
def add(sig, t, g):
    s = max(int(t * SR), 0)
    e = min(s + len(sig), len(out))
    out[s:e] += sig[:e - s] * g

w = whoosh()
for t in CUTS: add(w, t - 0.2, 0.14)
p = pop()
for t in POPS: add(p, t, 0.13)
for t in TICKS: add(tick(), t, 0.18)

pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())
