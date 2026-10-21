"""Builds the SFX bed (soft whoosh on every cut) for render.py's edit plan.

Usage: python3 audio.py OUT_SFX.wav
"""
import sys, wave
import numpy as np

SR = 44100
DUR = 73.1
# cut times = every B-roll start/end from render.py
CUTS = [3.80, 5.70, 6.75, 10.45, 11.25, 13.42, 16.45, 18.72, 21.38, 23.20, 26.70, 30.82, 34.02, 35.80,
        36.45, 39.28, 42.02, 44.00, 46.20, 48.70, 52.86, 56.90, 59.60, 63.20, 64.30, 68.55]

rng = np.random.default_rng(7)

def whoosh(d=0.36):
    n = int(SR * d)
    x = rng.normal(0, 1, n)
    # one-pole low-pass with a sweeping cutoff for an airy swish
    y = np.zeros(n); a = 0.0
    for i in range(n):
        c = 0.03 + 0.17 * np.sin(np.pi * i / n)
        a += c * (x[i] - a); y[i] = a
    env = np.sin(np.pi * np.arange(n) / n) ** 2
    return y * env / np.abs(y).max()

out = np.zeros(int(SR * DUR))
w = whoosh()
for t in CUTS:
    s = int((t - 0.22) * SR)
    out[s:s + len(w)] += w * 0.16

pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())
