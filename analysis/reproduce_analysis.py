#!/usr/bin/env python3
"""
Reproduces every number reported in Section 3 (Results) of the manuscript.

Usage:  python reproduce_analysis.py <folder_with_bin_files, e.g. ../data> [output_folder]

Input : 18 raw recordings  <gesture>_gesture_raw_<n>.bin  (gesture in fist | thumb_up | fingers_spread,
        n in 1,2,3,4,5,7 -> participants S1..S6), big-endian 16-bit words, ADS1015 12-bit left-justified.
Config: ADS1015 single-ended AIN0-GND (register 0x4283), PGA +-4.096 V -> 2 mV per count,
        sensor gain 1000 -> input-referred value = value at ADC output / 1000.
"""
import sys, os, json
import numpy as np, pandas as pd, scipy.signal as sg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis_out"
os.makedirs(OUT, exist_ok=True)

FS, LSB_MV, GAIN = 1000.0, 2.0, 1000.0
PART = {1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "S5", 7: "S6"}
GESTS = ["fist", "thumb_up", "fingers_spread"]
LABEL = {"fist": 0, "thumb_up": 1, "fingers_spread": 2}

sos_bp = sg.butter(4, [20, 450], btype="band", fs=FS, output="sos")
b_n, a_n = sg.iirnotch(50, 30, FS)


def find_file(name):
    """Look in DATA and in its sub-folders (data/S1 ... data/S6)."""
    for root, _, files in os.walk(DATA):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def load_counts(g, n):
    r = np.fromfile(find_file(f"{g}_gesture_raw_{n}.bin"), dtype=">u2").astype(int)
    return (r >> 4).astype(float)


def dsp(counts):
    """Zero-phase 20-450 Hz band-pass (4th-order Butterworth, 8 poles) + 50 Hz notch (Q=30). Output: mV at ADC."""
    x = counts * LSB_MV
    y = sg.sosfiltfilt(sos_bp, x - np.mean(x))
    return sg.filtfilt(b_n, a_n, y)


def lp_env(y, fc):
    return sg.sosfiltfilt(sg.butter(2, fc, fs=FS, output="sos"), np.abs(y))


# ---------------------------------------------------------------- A. per-recording signal metrics
rows = []
for g in GESTS:
    for n, p in PART.items():
        c = load_counts(g, n)
        y = dsp(c)
        t = np.arange(c.size) / FS
        e = lp_env(y, 1.0)
        m = t > 4
        thr = np.mean(np.percentile(e[m], [15, 85]))
        on = e > thr
        k = int(0.4 * FS)
        guard = np.zeros_like(on)
        for i in np.flatnonzero(np.diff(on.astype(int)) != 0):
            guard[max(0, i - k): i + k] = True
        A, R = on & ~guard & m, ~on & ~guard & m
        ra, rr = np.sqrt(np.mean(y[A] ** 2)), np.sqrt(np.mean(y[R] ** 2))
        # script segmentation (3 s + 5 s of each 8 s cycle): energy contrast
        seg_a = np.sqrt(np.mean(np.concatenate([y[int((r * 8 + 3) * FS):int((r * 8 + 8) * FS)] for r in range(10)]) ** 2))
        seg_r = np.sqrt(np.mean(np.concatenate([y[int(r * 8 * FS):int((r * 8 + 3) * FS)] for r in range(10)]) ** 2))
        rows.append(dict(
            participant=p, gesture=g, dc_V=c.mean() * LSB_MV / 1000,
            rail_low=int((c <= 2).sum()), rail_high=int((c >= 1495).sum()),
            n_bursts=int((np.diff(on[m].astype(int)) == 1).sum()),
            rest_rms_uV_in=rr / GAIN * 1000, active_rms_uV_in=ra / GAIN * 1000,
            snr_dB=20 * np.log10(ra / rr), rms_ratio_script_windows=seg_a / seg_r))
rec = pd.DataFrame(rows)
rec.to_csv(os.path.join(OUT, "per_recording.csv"), index=False, float_format="%.3f")
q_in_uV = LSB_MV / np.sqrt(12) / GAIN * 1000
print("\n=== A. Per-recording metrics ===")
print(rec.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\nIdeal quantisation noise (2 mV LSB): {LSB_MV/np.sqrt(12):.2f} mV rms at ADC output = {q_in_uV:.2f} uV input-referred")
sat = rec[(rec.rail_low > 0) | (rec.rail_high > 0)]
clean = rec.drop(rec[(rec.rail_low > 20) | (rec.rail_high > 20)].index)
print("\nRecordings touching a supply rail (>20 samples):",
      rec[(rec.rail_low > 20) | (rec.rail_high > 20)][["participant", "gesture", "rail_low", "rail_high"]].values.tolist())
for name, d in (("all 18", rec), ("excluding rail-saturated", clean)):
    print(f"SNR [{name}] mean {d.snr_dB.mean():.2f}  median {d.snr_dB.median():.2f}  min {d.snr_dB.min():.2f}  max {d.snr_dB.max():.2f}  n={len(d)}")
    print(f"rest RMS input-referred [{name}] median {d.rest_rms_uV_in.median():.1f} uV  min {d.rest_rms_uV_in.min():.1f}  max {d.rest_rms_uV_in.max():.1f}")
print("\nBy gesture (all 18):")
print(rec.groupby("gesture")[["snr_dB", "rest_rms_uV_in", "rms_ratio_script_windows"]].agg(["mean", "min", "max"]).round(2).to_string())
print("\nBy participant:")
print(rec.groupby("participant")[["snr_dB", "rest_rms_uV_in"]].mean().round(2).to_string())
print(f"\nRMS(3-8 s)/RMS(0-3 s) with the 8-s fixed grid: mean {rec.rms_ratio_script_windows.mean():.2f} "
      f"range {rec.rms_ratio_script_windows.min():.2f}-{rec.rms_ratio_script_windows.max():.2f}")


# ---------------------------------------------------------------- B. classification (authors' pipeline)
def feats(w):
    d = np.diff(w)
    return [np.mean(np.abs(w)), np.sqrt(np.mean(w ** 2)), np.var(w), np.sum(np.abs(d)),
            len(np.where(np.diff(np.sign(w)))[0]), len(np.where(np.diff(np.sign(d)))[0]), np.sum(np.abs(d) > 0.01)]


def windows(sig, part):
    X, rep = [], []
    for r in range(10):
        a, b = ((r * 8 + 3), (r * 8 + 8)) if part == "active" else ((r * 8), (r * 8 + 3))
        seg = sig[int(a * FS):int(b * FS)]
        for i in range(0, len(seg) - 200, 100):
            X.append(feats(seg[i:i + 200])); rep.append(r)
    return X, rep


def build(part, pairs):
    X, y, g, rp = [], [], [], []
    for gest, n in pairs:
        s = dsp(load_counts(gest, n))
        s = s / (np.max(np.abs(s)) + 1e-9)           # per-file max-abs normalisation (as in the original scripts)
        Xi, ri = windows(s, part)
        X += Xi; rp += ri; y += [LABEL[gest]] * len(Xi); g += [PART[n]] * len(Xi)
    return np.array(X), np.array(y), np.array(g), np.array(rp)


def rf():
    return RandomForestClassifier(200, random_state=42, n_jobs=-1)


def skf_acc(X, y, binary=False):
    if binary:
        m = y < 2; X, y = X[m], y[m]
    acc = [accuracy_score(y[te], rf().fit(X[tr], y[tr]).predict(X[te]))
           for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)]
    return 100 * np.mean(acc)


def grp_acc(X, y, groups, k):
    acc = [accuracy_score(y[te], rf().fit(X[tr], y[tr]).predict(X[te]))
           for tr, te in GroupKFold(k).split(X, y, groups)]
    return 100 * np.mean(acc)


res = {}
print("\n=== B1. Original experiment: fist_2, thumb_up_7, fingers_spread_3 (three different participants) ===")
orig = [("fist", 2), ("thumb_up", 7), ("fingers_spread", 3)]
X, y, g, rp = build("active", orig)
print("windows:", len(X))
res["orig_3class"] = skf_acc(X, y); res["orig_binary_fist_thumb"] = skf_acc(X, y, True)
cm = np.zeros((3, 3))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    cm += confusion_matrix(y[te], rf().fit(X[tr], y[tr]).predict(X[te]), labels=[0, 1, 2])
print(cm.astype(int))
Xr, yr, gr, rr_ = build("rest", orig)
res["orig_rest_only_negative_control"] = skf_acc(Xr, yr)
res["orig_active_grouped_by_repetition"] = grp_acc(X, y, rp, 5)
for k in ("orig_3class", "orig_binary_fist_thumb", "orig_rest_only_negative_control", "orig_active_grouped_by_repetition"):
    print(f"  {k}: {res[k]:.2f}%")

print("\n=== B2. All 18 recordings (6 participants x 3 gestures), same pipeline ===")
allp = [(gst, n) for gst in GESTS for n in PART]
Xa, ya, ga, ra = build("active", allp)
Xn, yn, gn, rn = build("rest", allp)
print("windows active:", len(Xa), " rest:", len(Xn))
res["pooled_random_5fold_active"] = skf_acc(Xa, ya)
res["pooled_random_5fold_rest_only"] = skf_acc(Xn, yn)
res["leave_one_participant_out_active"] = grp_acc(Xa, ya, ga, 6)
res["leave_one_participant_out_rest_only"] = grp_acc(Xn, yn, gn, 6)
within_r, within_g, within_n = [], [], []
for p in PART.values():
    m, mn = ga == p, gn == p
    within_r.append(skf_acc(Xa[m], ya[m]))
    within_g.append(grp_acc(Xa[m], ya[m], ra[m], 5))
    within_n.append(skf_acc(Xn[mn], yn[mn]))
res["within_participant_random_5fold_active"] = float(np.mean(within_r))
res["within_participant_grouped_by_rep_active"] = float(np.mean(within_g))
res["within_participant_random_5fold_rest_only"] = float(np.mean(within_n))
res["per_participant_random"] = [round(v, 1) for v in within_r]
res["per_participant_grouped_by_rep"] = [round(v, 1) for v in within_g]
res["per_participant_rest_only"] = [round(v, 1) for v in within_n]
for k, v in res.items():
    print(f"  {k}: {v if isinstance(v, list) else round(v, 2)}")
json.dump(res, open(os.path.join(OUT, "classification.json"), "w"), indent=2)
print("\nSaved:", os.path.join(OUT, "per_recording.csv"), "and", os.path.join(OUT, "classification.json"))
