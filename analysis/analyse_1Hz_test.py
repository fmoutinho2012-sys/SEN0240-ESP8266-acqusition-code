#!/usr/bin/env python3
"""
Analysis of the digitised 1 Hz test (data/tests/Test_1Hz_raw.bin).

Setup (paper, Section 2.5, item 2): FY3200S function generator, sine, 1 Hz, 1.5 Vpp, offset +0.75 V (so 0 to 1.5 V), applied to the
ADS1015 input (AIN0, single-ended, +-4.096 V, 2 mV per code). The usual firmware (1000 reads per second, 100-sample blocks written to
flash) records 80 s = 80 000 samples = 160 000 bytes (big-endian 16-bit words, conversion result in the upper 12 bits).

The script reports
  1. completeness and range: number of samples, minimum and maximum code, saturation;
  2. a sine fit (offset, amplitude, frequency, phase) to the whole record and to t >= 3 s. The 100-sample blocks are NOT equally spaced
     in time (see 4), so these fits leave a large residual and slightly underestimate the amplitude;
  3. a fit to the steady part of each block (positions 60-99), in which each block is allowed its own time shift. This gives amplitude and
     offset that are not distorted by the block timing, the residual of the steady part (quantisation step 2 mV -> 0.58 mV rms) and
     the mean residual in six level bins (linearity);
  4. the timing of the reads, seen through the 1 Hz sine: the lag of the samples relative to the steady part, by position in the block,
     the extra time at each block boundary (jump of the signal between the last sample of a block and the first of the next, divided by the
     local slope of the sine, minus the nominal 1 ms) and the blocks whose time shift changes by more than 3 ms.
     Only parts of the sine with a slope above 2 mV/ms are used (a 2 mV code step then corresponds to < 1 ms).

Usage: python analyse_1Hz_test.py <Test_1Hz_raw.bin> [output_folder]
Outputs: tone_1Hz_summary.json and (if matplotlib is installed) fig_1Hz_test.png
Set values of the generator: offset 750 mV, amplitude 750 mV, frequency 1 Hz (datasheet uncertainties +-1.5 % and +-50 ppm, not verified).
"""
import sys, os, json
import numpy as np
from scipy.optimize import least_squares

path = sys.argv[1] if len(sys.argv) > 1 else "Test_1Hz_raw.bin"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis_out"
os.makedirs(OUT, exist_ok=True)
LSB_MV, BLOCK, FS = 2.0, 100, 1000.0
SET_OFF, SET_AMP, SET_F = 750.0, 750.0, 1.0
MIN_SLOPE, MAX_RES = 2.0, 8.0          # mV/ms; mV rms of a block fit accepted as one time shift

raw = np.fromfile(path, dtype=">u2")
codes = raw.astype(int) >> 4
v = codes * LSB_MV
N = len(v); t = np.arange(N) / FS; B = N // BLOCK
pos = np.arange(N) % BLOCK
res = dict(n_samples=int(N), file_bytes=int(raw.nbytes), low_nibble_all_zero=bool(not (raw & 0xF).any()),
           min_code=int(codes.min()), max_code=int(codes.max()), min_mV=float(v.min()), max_mV=float(v.max()),
           mean_mV=float(v.mean()), saturated_samples=int((codes >= 2047).sum() + (codes < 0).sum()))

def sine_fit(mask):
    def f(a): return a[0] + a[1] * np.sin(2 * np.pi * a[2] * t[mask] + a[3]) - v[mask]
    best = min((least_squares(f, [v.mean(), 725, 1.0, q]) for q in np.linspace(0, 2 * np.pi, 8, endpoint=False)), key=lambda r: r.cost)
    a = best.x.copy()
    if a[1] < 0: a[1], a[3] = -a[1], a[3] + np.pi
    return a, f(a)

for name, m in (("whole_record", np.ones(N, bool)), ("t_ge_3s", t >= 3)):
    a, r = sine_fit(m)
    res["fit_" + name] = dict(offset_mV=float(a[0]), amplitude_mV=float(a[1]), peak_to_peak_mV=float(2 * a[1]), frequency_Hz=float(a[2]),
                              residual_rms_mV=float(r.std()))
a, _ = sine_fit(t >= 3)
O, A, F = a[0], a[1], a[2]
PH = a[3]

# --- steady part of each block with a free time shift per block --------------------------------------------------------------
sel = pos >= 60
def shifts_for(O, A, F, PH):
    sh = np.zeros(B); cost = np.zeros(B); slope = np.zeros(B)
    for b in range(B):
        s = slice(b * BLOCK + 60, (b + 1) * BLOCK)
        g = lambda x: O + A * np.sin(2 * np.pi * F * (t[s] + x[0]) + PH) - v[s]
        best = min((least_squares(g, [x0], bounds=(-0.06, 0.2)) for x0 in (-0.01, 0.0, 0.03, 0.08, 0.15)), key=lambda r: r.cost)
        sh[b] = best.x[0]; cost[b] = np.sqrt(2 * best.cost / (BLOCK - 60))
        slope[b] = abs(A * 2 * np.pi * F * np.cos(2 * np.pi * F * (t[s].mean() + sh[b]) + PH)) / 1000.0
    return sh, cost, slope
sh = np.zeros(B)
for it in range(4):
    X = np.c_[np.sin(2 * np.pi * F * (t + np.repeat(sh, BLOCK)) + PH), np.ones(N)]
    (A, O), *_ = np.linalg.lstsq(X[sel], v[sel], rcond=None)
    sh, cost, slope = shifts_for(O, A, F, PH)
steady_resid = (O + A * np.sin(2 * np.pi * F * (t + np.repeat(sh, BLOCK)) + PH) - v)
ok = (slope > MIN_SLOPE) & (cost < MAX_RES)
base = np.median(sh[ok]); dev = (sh - base) * 1000.0                    # ms, > 0: later than the steady part
good = ok & (np.abs(dev) < 3)
gm = np.repeat(good, BLOCK) & sel
mdl = O + A * np.sin(2 * np.pi * F * (t + np.repeat(sh, BLOCK)) + PH)
lev = mdl[gm]; rr = (v - mdl)[gm]; bins = np.digitize(lev, np.linspace(0, 1500, 7))
tb = np.arange(B) * 0.1 + 0.08
cf = np.polyfit(tb[good], sh[good], 1)
res["steady_part_fit"] = dict(
    offset_mV=float(O), amplitude_mV=float(A), peak_to_peak_mV=float(2 * A),
    offset_error_mV=float(O - SET_OFF), amplitude_error_percent=float(100 * (A - SET_AMP) / SET_AMP),
    residual_rms_mV_positions_60_99_all_blocks=float(steady_resid[sel].std()),
    residual_rms_mV_steady_blocks=float(rr.std()),
    mean_residual_mV_by_level_bin_0_1500mV=[float(rr[bins == k].mean()) for k in range(1, 7)],
    drift_of_block_time_offset_s_per_s=float(cf[0]),
    apparent_frequency_Hz=float(F * (1 + cf[0])))
res["whole_record_errors_vs_set_values"] = dict(
    offset_error_mV=res["fit_whole_record"]["offset_mV"] - SET_OFF,
    amplitude_error_percent=100 * (res["fit_whole_record"]["amplitude_mV"] - SET_AMP) / SET_AMP,
    frequency_error_ppm=1e6 * (res["fit_whole_record"]["frequency_Hz"] - SET_F) / SET_F)

# --- timing of the reads --------------------------------------------------------------------------------------------------
der = lambda tt: A * 2 * np.pi * F * np.cos(2 * np.pi * F * (tt + base) + PH) / 1000.0     # mV/ms
mod = lambda tt: O + A * np.sin(2 * np.pi * F * (tt + base) + PH)
vv = v.reshape(B, BLOCK); tt = t.reshape(B, BLOCK)
prof = []
for q in range(0, BLOCK, 10):
    ss = []
    for b in np.where(good)[0]:
        x, y = tt[b, q:q + 10], vv[b, q:q + 10]; d = der(x); m = np.abs(d) > MIN_SLOPE
        ss.extend(((y - mod(x)) / d)[m])
    prof.append(dict(positions=f"{q}-{q + 9}", median_lag_ms=float(np.median(ss)), q25=float(np.percentile(ss, 25)),
                     q75=float(np.percentile(ss, 75)), n=len(ss)))
ex = []
for b in np.where(good[1:] & good[:-1])[0]:
    d = der(tt[b, -1])
    if abs(d) > MIN_SLOPE: ex.append((vv[b + 1, 0] - vv[b, -1]) / d - 1.0)
ex = np.array(ex)
dsh = np.diff(sh * 1000.0); m2 = ok[:-1] & ok[1:]
jumps = np.sort(dsh[m2][dsh[m2] > 3])
res["block_timing"] = dict(
    note="lag of the samples relative to the steady part of the block (positions 60-99); blocks with a steady time shift only",
    lag_by_position=prof, n_steady_blocks=int(good.sum()), n_blocks_with_steep_slope=int((slope > MIN_SLOPE).sum()),
    n_blocks_not_fitted_by_one_shift=int(((slope > MIN_SLOPE) & (cost >= MAX_RES)).sum()),
    boundary_extra_time_ms_median=float(np.median(ex)), boundary_extra_time_ms_q25_q75=[float(x) for x in np.percentile(ex, [25, 75])],
    n_boundaries=int(len(ex)),
    first_blocks_lag_ms=[float(x) for x in dev[:12]],
    boundary_jumps_in_time_shift_over_3ms=[float(x) for x in jumps])
# --- long stalls: groups of consecutive blocks whose time shift exceeds the steady value by more than 30 ms ---------------------
# (the blocks of a stall are not accepted by the one-shift fit above, because the lag changes inside them)
ev = np.where(dev > 30)[0]
groups = np.split(ev, np.where(np.diff(ev) > 2)[0] + 1) if len(ev) else []
groups = [g for g in groups if len(g)]
start_blocks = np.array([g[0] for g in groups]); peaks = np.array([dev[g].max() for g in groups])
later = start_blocks > 10                                                    # leave out the start-up stall of the first blocks
res["long_stalls"] = dict(
    definition="groups of blocks with time shift > 30 ms above the steady value (positions 60-99); lag peak = largest block value",
    n_groups_after_start=int(later.sum()), start_up_peak_ms=float(peaks[~later].max()) if (~later).any() else None,
    peak_lag_ms_median=float(np.median(peaks[later])), peak_lag_ms_min_max=[float(peaks[later].min()), float(peaks[later].max())],
    spacing_blocks_median=float(np.median(np.diff(start_blocks[later]))),
    spacing_s_median=float(np.median(np.diff(start_blocks[later]))) * BLOCK / FS,
    blocks_per_4096_bytes=float(4096 / (BLOCK * 2)),
    mean_spacing_s_over_record=float((start_blocks[later][-1] - start_blocks[later][0]) / (later.sum() - 1) * BLOCK / FS))
json.dump(res, open(os.path.join(OUT, "tone_1Hz_summary.json"), "w"), indent=2)
print(json.dumps(res, indent=1)[:6000])

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    k = slice(0, 3000)
    ax[0].plot(t[k], v[k], lw=.6, label="digitised"); ax[0].plot(t[k], (O + A * np.sin(2 * np.pi * F * t + PH + 2 * np.pi * F * base))[k],
                                                             lw=.6, label="sine, steady-part fit"); ax[0].legend(fontsize=7)
    ax[0].set(xlabel="time (s)", ylabel="mV at the converter", title="(a) first 3 s; steps every 0.1 s = block writes")
    xs = [int(p["positions"].split("-")[0]) + 4.5 for p in prof]
    ax[1].errorbar(xs, [p["median_lag_ms"] for p in prof], yerr=[[p["median_lag_ms"] - p["q25"] for p in prof], [p["q75"] - p["median_lag_ms"] for p in prof]], fmt="o-")
    ax[1].set(xlabel="position in the 100-sample block", ylabel="lag of the samples (ms)", title="(b) median and IQR, steady blocks")
    ax[2].plot(np.arange(B) * 0.1, dev, lw=.6); ax[2].set(xlabel="time (s)", ylabel="block time shift (ms)", ylim=(-10, 140),
                                                           title="(c) time shift of each block (positions 60-99)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_1Hz_test.png"), dpi=140)
except Exception as e:
    print("figure skipped:", e)
