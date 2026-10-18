#!/usr/bin/env python3
"""
Analysis of the digitised 250 Hz test tone (data/tests/Test_250Hz_raw.bin).

Setup (see the paper, Section 2.5): a 250 Hz sine of 1.0 Vpp with +1.5 V offset from a function generator is applied to the
ADS1015 input, and the usual firmware (1000 reads per second, 100-sample blocks written to flash) records 80 s.
At 1000 reads per second a 250 Hz tone has exactly 4 samples per cycle, so pairs of neighbouring samples are in quadrature
and the step between samples can be turned into a phase step; a phase step of 90 degrees means a read interval of 1 ms.

Usage: python analyse_250Hz_test.py <Test_250Hz_raw.bin> [output_folder]
Outputs: test_250Hz_summary.json and (if matplotlib is installed) fig_250Hz_test.png
"""
import sys, os, json
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "Test_250Hz_raw.bin"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis_out"
os.makedirs(OUT, exist_ok=True)
FS, LSB_MV, F0, BLOCK = 1000.0, 2.0, 250.0, 100

v = (np.fromfile(path, dtype=">u2").astype(int) >> 4) * LSB_MV      # mV at the converter input
N = len(v)
u = v - v.mean()
res = dict(n_samples=int(N), mean_mV=float(v.mean()), amplitude_from_std_mV=float(np.sqrt(2) * u.std()),
           peak_to_peak_mV=float(v.max() - v.min()))
res["vpp_estimate_mV"] = 2 * res["amplitude_from_std_mV"]

# Spectrum (rectangular window; 250 Hz is an exact bin for 80 000 samples)
X = np.fft.rfft(u); P = np.abs(X) ** 2; f = np.fft.rfftfreq(N, 1 / FS)
k = int(round(F0 * N / FS)); tot = P[1:].sum()
for w, name in ((1, "line_pm1bin"), (3, "line_pm3bins")):
    line = P[k - w:k + w + 1].sum()
    res[f"power_fraction_{name}"] = float(line / tot)
    res[f"coherent_snr_dB_{name}"] = float(10 * np.log10(line / (tot - line)))
res["power_fraction_240_260Hz"] = float(P[int(240 * N / FS):int(260 * N / FS)].sum() / tot)
Ps = np.convolve(P, np.ones(8) / 8, "same")      # light smoothing for line levels
lines = {}
for fr in (170, 180, 190, 200, 240, 260):
    j = int(round(fr * N / FS)); lines[str(fr)] = float(10 * np.log10(P[j - 1:j + 2].sum() / P[k - 1:k + 2].sum()))
res["line_levels_dB_re_250Hz"] = lines

# Phase step between neighbouring samples (quadrature pairs); deviation from 90 degrees
th = np.arctan2(u[:-1], u[1:])
d = np.angle(np.exp(1j * (th[1:] - th[:-1] - np.pi / 2)))        # rad; 1.571 rad = 1 ms at 250 Hz
pos = np.arange(len(d)) % BLOCK
short = d < -0.85                                                 # step shorter by more than about 0.54 ms
res["fraction_short_steps"] = float(short.mean())
frac_pos = np.array([short[pos == q].mean() for q in range(BLOCK)])
res["short_step_fraction_by_position_in_block"] = [round(float(x), 3) for x in frac_pos]
away = (pos >= 40) & (pos <= 90)
res["timing_sigma_per_sample_away_from_block_start_us"] = float(d[away].std() / (np.sqrt(2) * 2 * np.pi * F0) * 1e6)
res["note"] = ("The timing sigma assumes independent timing errors of consecutive samples; the 1.0 ms nominal step and "
               "uniform conversion window of 0.625 ms give 180 us.")
json.dump(res, open(os.path.join(OUT, "test_250Hz_summary.json"), "w"), indent=2)
for key, val in res.items():
    if "by_position" not in key:
        print(f"{key}: {val}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    import scipy.signal as sg
    fw, Pw = sg.welch(u, fs=FS, window="hann", nperseg=4000)      # 4 s segments, 0.25 Hz resolution
    db = 10 * np.log10(Pw / Pw.max() + 1e-12)
    ax[0].plot(fw[(fw > 0) & (fw < 500)], db[(fw > 0) & (fw < 500)], lw=.8, color="#1f4e79")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="Level re the 250 Hz line (dB)", ylim=(-70, 5), title="(a) Spectrum of the digitised tone (4 s segments)")
    ax[1].bar(np.arange(BLOCK), 100 * frac_pos, color="#1f4e79", width=.9)
    ax[1].set(xlabel="Position of the step in the 100-sample block", ylabel="Steps shorter by > 0.5 ms (%)", title="(b) Short read intervals by block position")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_250Hz_test.png"), dpi=200)
except ImportError:
    pass
