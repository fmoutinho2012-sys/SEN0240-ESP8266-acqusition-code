#!/usr/bin/env python3
"""
How much do the read-timing irregularities of the firmware change the time-domain and spectral features used in the paper?

There is no ground truth for the recorded sEMG, so this is a simulation. A stationary Gaussian signal whose power spectrum is the
average of the 18 band-passed recordings is generated at 16 kHz. It is sampled (a) ideally, every 1 ms, and (b) with the timing
inferred from the test tones (paper, Section 3.1):
  (b1) steady state only: each read returns the latest conversion of a converter running at 1600 SPS (instant uniform in a
       0.625 ms window);
  (b2) steady state plus the block structure: at the start of every 100-sample block the loop is late by L ms (pause for writing
       the block to flash) and reads every 0.73 ms until it has caught up (lag decreases by 0.27 ms per read). L is drawn so that the
       fraction of blocks still catching up at position k follows the median read interval measured with the 100 Hz tone.
The same filter chain and the same features as reproduce_analysis.py are applied to each version, on 200-sample windows with a
100-sample step, and the windows of (a) and (b) are compared.

Usage: python simulate_timing_effect.py <data_folder> <tone_100Hz_file> [output_folder] [n_seeds]   (the paper uses 20 seeds)
"""
import sys, os, json
import numpy as np, scipy.signal as sg
from scipy.interpolate import CubicSpline

DATA, TONE = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "analysis_out"
NSEED = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(OUT, exist_ok=True)
FS, LSB, FH, DUR = 1000.0, 2.0, 16000.0, 80.0
sos_bp = sg.butter(4, [20, 450], btype="band", fs=FS, output="sos"); b_n, a_n = sg.iirnotch(50, 30, FS)

def dsp(x):
    y = sg.sosfiltfilt(sos_bp, x - np.mean(x)); return sg.filtfilt(b_n, a_n, y)

def feats(w):
    d = np.diff(w)
    mag = np.abs(np.fft.rfft(w * np.hanning(len(w)), 1024)) ** 2; f = np.fft.rfftfreq(1024, 1 / FS)
    c = np.cumsum(mag)
    return [np.mean(np.abs(w)), np.sqrt(np.mean(w ** 2)), np.var(w), np.sum(np.abs(d)),
            len(np.where(np.diff(np.sign(w)))[0]), len(np.where(np.diff(np.sign(d)))[0]), np.sum(np.abs(d) > 0.01),
            np.sum(f * mag) / np.sum(mag), f[np.searchsorted(c, c[-1] / 2)]]
NAMES = ["MAV", "RMS", "VAR", "WL", "ZC", "SSC", "WAMP", "MNF", "MDF"]

def window_feats(s):
    s = s / (np.max(np.abs(s)) + 1e-9)                   # per-record max-abs normalisation, as in the paper's pipeline
    return np.array([feats(s[i:i + 200]) for i in range(0, len(s) - 200, 100)])

# 1. target spectrum: mean of the normalised PSDs of the 18 recordings
psds = []
for root, _, files in os.walk(DATA):
    for fn in files:
        if fn.endswith(".bin") and "gesture_raw" in fn:
            x = (np.fromfile(os.path.join(root, fn), dtype=">u2").astype(int) >> 4) * LSB
            f, p = sg.welch(dsp(x), fs=FS, nperseg=1024); psds.append(p / p.sum())
psd = np.mean(psds, 0); print("recordings used:", len(psds))

# 2. fraction of blocks still catching up at position k, from the median read interval of the 100 Hz tone
v = (np.fromfile(TONE, dtype=">u2").astype(int) >> 4) * LSB; u = v - v.mean()
dt = np.diff(np.unwrap(np.angle(sg.hilbert(u)))) / (2 * np.pi * 100) * 1e3
pos = np.arange(len(dt)) % 100
med = sg.medfilt(np.array([np.median(dt[pos == q]) for q in range(100)]), 5)
p_k = np.minimum.accumulate(np.clip((1.0 - med) / (1.0 - 0.73), 0, 1)); p_k[:3] = 1.0; p_k[96:] = 0
def draw_L(rng):                                          # lag at the block start, ms
    uu = rng.random(); k = np.searchsorted(-p_k, -uu); return 0.27 * k

def sample_times(rng, mode, n):
    t = np.arange(n, dtype=float)                         # ms
    if mode in ("steady", "full"):
        t = t.copy()
    if mode == "full":
        for b in range(n // 100):
            L = draw_L(rng); k = np.arange(100)
            t[b * 100:(b + 1) * 100] += np.maximum(L - 0.27 * k, 0)
    ph = rng.random()                                      # conversions every 0.625 ms; read returns the latest one
    return (np.floor(t * 1.6 - ph) + ph) / 1.6

# 3. model check: coherent SNR of simulated tones against the measured values of the test tones
check = {}
for f0, meas in ((50, -4.59), (100, -5.07), (250, -6.05), (450, -6.99)):
    vals = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed); n = int(DUR * FS)
        x = np.sin(2 * np.pi * f0 * sample_times(rng, "full", n) * 1e-3 + rng.random() * 6.28)
        P = np.abs(np.fft.rfft(x - x.mean())) ** 2; k = int(round(f0 * n / FS)); line = P[k - 1:k + 2].sum()
        vals.append(10 * np.log10(line / (P[1:].sum() - line)))
    check[str(f0)] = dict(simulated_dB=round(float(np.mean(vals)), 1), measured_dB=meas)
print("model check (coherent SNR, simulated vs measured):", check)

res = {}
for seed in range(NSEED):
    rng = np.random.default_rng(seed)
    nh = int(DUR * FH); white = rng.standard_normal(nh)
    F = np.fft.rfft(white); fr = np.fft.rfftfreq(nh, 1 / FH)
    gain = np.sqrt(np.interp(fr, np.fft.rfftfreq(1024, 1 / FS), psd, right=0))
    hi = np.fft.irfft(F * gain, nh) * 200.0               # arbitrary amplitude (features are normalised)
    cs = CubicSpline(np.arange(nh) / FH * 1e3, hi)
    n = int(DUR * FS)
    base = window_feats(dsp(cs(np.arange(n, dtype=float))))
    for mode in ("steady", "full"):
        tt = sample_times(rng, mode, n); tt = np.clip(tt, 0, nh / FH * 1e3 - 1)
        w = window_feats(dsp(cs(tt)))
        ratio = w / base
        res.setdefault(mode, []).append(dict(
            bias=[float(np.mean(w[:, i]) / np.mean(base[:, i]) - 1) for i in range(len(NAMES))],
            scatter=[float(np.std(ratio[:, i])) for i in range(len(NAMES))]))
out = {}
for mode in res:
    b = np.array([r["bias"] for r in res[mode]]); s = np.array([r["scatter"] for r in res[mode]])
    out[mode] = {NAMES[i]: dict(bias_pct=round(100 * float(b[:, i].mean()), 2), bias_sd_pct=round(100 * float(b[:, i].std()), 2),
                                window_scatter_pct=round(100 * float(s[:, i].mean()), 2)) for i in range(len(NAMES))}
out["p_still_catching_up_by_position"] = [round(float(x), 2) for x in p_k]
out["model_check_tone_snr"] = check
out["note"] = "Simulation with an inferred timing model; not a measurement on the recordings. Bias = change in the mean of the feature over all windows; scatter = standard deviation of the ratio window by window."
json.dump(out, open(os.path.join(OUT, "timing_effect_summary.json"), "w"), indent=2)
for mode in ("steady", "full"):
    print(mode); [print(f"  {k}: bias {v['bias_pct']:+.1f} % (sd {v['bias_sd_pct']:.1f}), window scatter {v['window_scatter_pct']:.1f} %") for k, v in out[mode].items()]
