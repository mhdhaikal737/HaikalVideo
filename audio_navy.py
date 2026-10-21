"""Builds the SFX bed for render_navy.py from the events it logged.

Usage: python3 audio_navy.py SFX.json OUT_SFX.wav
"""
import sys, json, wave
import numpy as np

SR = 44100
DUR = 73.1
rng = np.random.default_rng(7)

def env(n, a=0.004, d=8.0):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t * d)

def pop(d=0.09):
    n = int(SR * d); t = np.arange(n) / SR
    f = 1000 * np.exp(-t * 18) + 420
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 38)

def click():
    n = int(SR * 0.03); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2400 * t) * 0.6 + rng.normal(0, 0.3, n)) * env(n, 0.0005, 160)

def ding():
    n = int(SR * 0.9); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1318 * t) + 0.6 * np.sin(2 * np.pi * 1976 * t)) * env(n, 0.002, 5) * 0.6

def shine():
    n = int(SR * 0.5); t = np.arange(n) / SR
    f = 2600 + 1800 * t / 0.5
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.01, 9) * 0.35

def boom():
    n = int(SR * 0.7); t = np.arange(n) / SR
    f = 120 * np.exp(-t * 5) + 45
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.003, 5)

def whoosh(d=0.34):
    n = int(SR * d)
    x = rng.normal(0, 1, n)
    y = np.zeros(n); a = 0.0
    for i in range(n):
        c = 0.03 + 0.2 * np.sin(np.pi * i / n)
        a += c * (x[i] - a); y[i] = a
    return y * np.sin(np.pi * np.arange(n) / n) ** 2 / np.abs(y).max()

BANK = {'pop': (pop(), 0.22, 0.0), 'click': (click(), 0.3, 0.0), 'ding': (ding(), 0.25, 0.0),
        'shine': (shine(), 0.2, 0.0), 'boom': (boom(), 0.5, 0.0), 'whoosh': (whoosh(), 0.2, -0.2)}

out = np.zeros(int(SR * DUR) + SR)
last = {}
for kind, t in json.load(open(sys.argv[1])):
    if t - last.get(kind, -9) < 0.12: continue  # avoid machine-gun stacking
    last[kind] = t
    snd, gain, off = BANK[kind]
    s = int(max(0, t + off) * SR)
    out[s:s + len(snd)] += snd * gain

pcm = (np.clip(out[:int(SR * DUR)], -1, 1) * 32767).astype(np.int16)
with wave.open(sys.argv[2], 'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())
