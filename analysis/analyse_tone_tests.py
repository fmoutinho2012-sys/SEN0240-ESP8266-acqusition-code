#!/usr/bin/env python3
"""
Analysis of the digitised test tones (data/tests/Test_50Hz_raw.bin, Test_100Hz_raw.bin, Test_250Hz_raw.bin, Test_450Hz_raw.bin).

Setup (paper, Section 2.5): a sine of 1.0 Vpp with +1.5 V offset from a function generator is applied to the ADS1015 input and the
usual firmware (1000 reads per second, 100-sample blocks written to flash) records 80 s (80 000 samples).

For each tone the script reports
  - mean and amplitude (from the standard deviation) at the converter input;
  - the fraction of the power in the tone line (+-1 bin of 0.0125 Hz) and the coherent SNR = line / (everything else);
  - the first-order ceiling from conversion timing, -20*log10(2*pi*f*sigma_t), sigma_t = 0.625 ms/sqrt(12) = 0.18 ms;
  - for 1 s windows, how often the strongest line is the tone and how often it lies at a lower frequency (0.6-0.9 f0),
    and the median ratio of that lower frequency to f0 (a ratio of 0.73 means samples spaced 0.73 ms apart instead of 1 ms);
  - the timing scatter per read in the steady part of the block (positions 60-95), from a sine fit to each block;
  - for tones of 100 Hz or less, the median read interval by position in the 100-sample block, from the phase of the analytic
    signal (approximate; intervals longer than half a period of the tone cannot be unwrapped, so the boundary step is not measured).

Usage: python analyse_tone_tests.py <folder_with_Test_*Hz_raw.bin> [output_folder]
Outputs: tone_tests_summary.json and (if matplotlib is installed) fig_tone_tests.png
"""
import sys, os, json
import numpy as np
import scipy.signal as sg

folder = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis_out"
os.makedirs(OUT, exist_ok=True)
FS, LSB_MV, BLOCK, SIGMA_T = 1000.0, 2.0, 100, 0.625e-3 / np.sqrt(12)
TONES = (50, 100, 250, 450)

res, prof = {}, {}
for f0 in TONES:
    p = os.path.join(folder, f"Test_{f0}Hz_raw.bin")
    if not os.path.exists(p):
        continue
    v = (np.fromfile(p, dtype=">u2").astype(int) >> 4) * LSB_MV           # mV at the converter input
    N = len(v); u = v - v.mean()
    P = np.abs(np.fft.rfft(u)) ** 2
    k = int(round(f0 * N / FS)); tot = P[1:].sum(); line = P[k - 1:k + 2].sum()
    r = dict(n_samples=int(N), mean_mV=float(v.mean()), amplitude_from_std_mV=float(np.sqrt(2) * u.std()),
             power_fraction_line=float(line / tot), coherent_snr_dB=float(10 * np.log10(line / (tot - line))),
             timing_ceiling_dB=float(-20 * np.log10(2 * np.pi * f0 * SIGMA_T)))
    n_tone = n_low = 0; ratios = []
    for s in range(N // 1000):
        x = u[s * 1000:(s + 1) * 1000] * np.hanning(1000)
        Pw = np.abs(np.fft.rfft(x)) ** 2
        j = int(np.argmax(Pw[5:495])) + 5
        if abs(j - f0) <= 2: n_tone += 1
        elif 0.6 * f0 <= j <= 0.9 * f0: n_low += 1; ratios.append(j / f0)
    r.update(windows_1s=int(N // 1000), windows_strongest_line_at_tone=n_tone, windows_strongest_line_below_tone=n_low,
             median_ratio_of_lower_line_to_tone=float(np.median(ratios)) if ratios else None)
    # timing scatter per read in the steady part of the block (positions 60-95): residual of a sine fit (amplitude, phase, offset)
    # to each block, quantisation noise (0.58 mV) removed, divided by the RMS slope of the tone
    Xb = v.reshape(-1, BLOCK); tt = np.arange(60, 96) * 1e-3
    Am = np.c_[np.cos(2 * np.pi * f0 * tt), np.sin(2 * np.pi * f0 * tt), np.ones(len(tt))]
    amps, rr = [], []
    for row in Xb:
        c, *_ = np.linalg.lstsq(Am, row[60:96], rcond=None)
        amps.append(np.hypot(c[0], c[1])); rr.append(((row[60:96] - Am @ c) ** 2).sum() / (len(tt) - 3))
    rms = np.sqrt(np.median(rr)); A = np.median(amps)
    r["steady_timing_scatter_us"] = float(np.sqrt(max(rms ** 2 - 0.58 ** 2, 0)) / (A * 2 * np.pi * f0 / np.sqrt(2)) * 1e6)
    if f0 <= 100:
        ph = np.unwrap(np.angle(sg.hilbert(u)))
        dt = np.diff(ph) / (2 * np.pi * f0) * 1e3                      # ms per step
        pos = np.arange(len(dt)) % BLOCK
        med = np.array([np.median(dt[pos == q]) for q in range(BLOCK)])
        prof[f0] = med
        r["median_step_ms_positions_0_39"] = float(np.median(med[:40]))
        r["median_step_ms_positions_40_98"] = float(np.median(med[40:99]))
    res[str(f0)] = r
res["note"] = ("Windows of 1 s, Hann. The mean read rate is 1000 per second (the tone peaks at its exact frequency in the 80 s spectrum), "
               "so the 0.7 ms intervals must be compensated by longer ones; these are not measurable with the 50 and 100 Hz tones "
               "(more than half a period) and are inferred from the constant mean rate.")
json.dump(res, open(os.path.join(OUT, "tone_tests_summary.json"), "w"), indent=2)
for f0 in TONES:
    if str(f0) in res:
        print(f0, "Hz:", {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in res[str(f0)].items()})

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    fr = np.array([f for f in TONES if str(f) in res])
    ax[0].plot(np.logspace(1.5, 2.8, 100), -20 * np.log10(2 * np.pi * np.logspace(1.5, 2.8, 100) * SIGMA_T), "--", color="0.4",
               label="ceiling from conversion timing (σt = 0.18 ms)")
    ax[0].plot(fr, [res[str(f)]["coherent_snr_dB"] for f in fr], "o-", color="#1f4e79", label="measured, digitised tone")
    ax[0].set(xscale="log", xlabel="Tone frequency (Hz)", ylabel="Coherent SNR (dB)", title="(a) Tone SNR after digitisation", xticks=fr, xticklabels=[str(f) for f in fr])
    ax[0].minorticks_off(); ax[0].legend(fontsize=7.5, frameon=False)
    show = np.ones(BLOCK, bool); show[[0, 1, 2, 96, 97, 98, 99]] = False     # steps next to the boundary cannot be unwrapped
    for f0_, col in ((100, "#1f4e79"), (50, "#c55a11")):
        if f0_ in prof:
            ax[1].plot(np.arange(BLOCK)[show], prof[f0_][show], ".-", color=col, ms=3, lw=.8, label=f"{f0_} Hz tone")
    ax[1].axhline(1.0, color="0.5", lw=.8, ls=":")
    ax[1].set(xlabel="Position in the 100-sample block", ylabel="Median read interval (ms)", ylim=(0.4, 1.4), title="(b) Read interval by block position (approximate)")
    ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_tone_tests.png"), dpi=200)
except ImportError:
    pass
