#!/usr/bin/env python3
"""
Re-analysis with data-driven segmentation.

The nominal protocol (ten 8 s cycles, 5 s contraction + 3 s rest) does not match the recordings: the envelope bursts
repeat about every 4.3 s. Fixed windows (3-8 s / 0-3 s of each 8 s cycle) therefore mix contraction and rest.
Here the contraction and rest segments come from the detector of reproduce_analysis.py (1 Hz envelope, threshold at
the midpoint of its 15th and 85th percentiles after 4 s, 0.4 s guard around transitions). Windows (200 ms, step 100 ms)
must lie entirely inside one segment. Same features and Random Forest as reproduce_analysis.py.

Usage: python reanalysis_detected_segments.py <folder_with_bin_files, e.g. ../data> [output_folder]
Outputs: period_per_recording.csv, detected_segments_classification.json
"""
import sys, os, json
import numpy as np, pandas as pd, scipy.signal as sg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis_out"
os.makedirs(OUT, exist_ok=True)
FS, LSB_MV = 1000.0, 2.0
PART = {1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "S5", 7: "S6"}
GESTS = ["fist", "thumb_up", "fingers_spread"]
LABEL = {g: i for i, g in enumerate(GESTS)}
sos_bp = sg.butter(4, [20, 450], btype="band", fs=FS, output="sos")
b_n, a_n = sg.iirnotch(50, 30, FS)
sos_1hz = sg.butter(2, 1.0, fs=FS, output="sos")


def find_file(name):
    for root, _, files in os.walk(DATA):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def load_counts(g, n):
    return (np.fromfile(find_file(f"{g}_gesture_raw_{n}.bin"), dtype=">u2").astype(int) >> 4).astype(float)


def dsp(c):
    x = c * LSB_MV
    return sg.filtfilt(b_n, a_n, sg.sosfiltfilt(sos_bp, x - x.mean()))


def detect(y):
    e = sg.sosfiltfilt(sos_1hz, np.abs(y))
    t = np.arange(y.size) / FS
    m = t > 4
    on = e > np.mean(np.percentile(e[m], [15, 85]))
    k = int(0.4 * FS)
    guard = np.zeros_like(on)
    for i in np.flatnonzero(np.diff(on.astype(int)) != 0):
        guard[max(0, i - k): i + k] = True
    return e, on & ~guard & m, ~on & ~guard & m


def period_s(e):
    x = sg.detrend(e[4000::10], type="linear")
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    lo, hi = 150, 1000
    pk, _ = sg.find_peaks(ac[lo:hi])
    return (pk[np.argmax(ac[lo:hi][pk])] + lo) / 100 if len(pk) else np.nan


def feats(w):
    d = np.diff(w)
    return [np.mean(np.abs(w)), np.sqrt(np.mean(w ** 2)), np.var(w), np.sum(np.abs(d)),
            len(np.where(np.diff(np.sign(w)))[0]), len(np.where(np.diff(np.sign(d)))[0]), np.sum(np.abs(d) > 0.01)]


def seg_windows(sig, mask):
    """Windows lying fully inside one contiguous True run; returns features and segment ids."""
    X, sid = [], []
    d = np.diff(np.r_[0, mask.astype(int), 0])
    for s_id, (a, b) in enumerate(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))):
        for i in range(a, b - 200 + 1, 100):
            X.append(feats(sig[i:i + 200])); sid.append(s_id)
    return X, sid


rows, per = [], []
data = {"active": [], "rest": []}
for g in GESTS:
    for n, p in PART.items():
        y = dsp(load_counts(g, n))
        e, A, R = detect(y)
        per.append(dict(participant=p, gesture=g, period_s=period_s(e)))
        s = y / (np.max(np.abs(y)) + 1e-9)
        for key, mask in (("active", A), ("rest", R)):
            X, sid = seg_windows(s, mask)
            for x, q in zip(X, sid):
                data[key].append((x, LABEL[g], p, f"{p}_{g}_{q}"))
pd.DataFrame(per).to_csv(os.path.join(OUT, "period_per_recording.csv"), index=False, float_format="%.2f")
pdf = pd.DataFrame(per)
print(f"Dominant period of the envelope (autocorrelation): median {pdf.period_s.median():.2f} s, "
      f"IQR {pdf.period_s.quantile(.25):.2f}-{pdf.period_s.quantile(.75):.2f}, range {pdf.period_s.min():.2f}-{pdf.period_s.max():.2f}")


def arrays(key):
    X = np.array([r[0] for r in data[key]]); y = np.array([r[1] for r in data[key]])
    pa = np.array([r[2] for r in data[key]]); sg_ = np.array([r[3] for r in data[key]])
    return X, y, pa, sg_


def rf():
    return RandomForestClassifier(200, random_state=42, n_jobs=-1)


def acc_skf(X, y):
    return 100 * np.mean([accuracy_score(y[te], rf().fit(X[tr], y[tr]).predict(X[te]))
                          for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)])


def acc_grp(X, y, g, k):
    return 100 * np.mean([accuracy_score(y[te], rf().fit(X[tr], y[tr]).predict(X[te]))
                          for tr, te in GroupKFold(k).split(X, y, g)])


res = {}
for key in ("active", "rest"):
    X, y, pa, sgm = arrays(key)
    res[f"{key}_n_windows"] = int(len(X))
    res[f"{key}_majority_class_baseline"] = 100 * np.bincount(y).max() / len(y)
    res[f"{key}_pooled_random_5fold"] = acc_skf(X, y)
    res[f"{key}_leave_one_participant_out"] = acc_grp(X, y, pa, 6)
    res[f"{key}_lopo_per_participant"] = [round(100 * accuracy_score(y[pa == p], rf().fit(X[pa != p], y[pa != p]).predict(X[pa == p])), 1)
                                          for p in PART.values()]
    w_r, w_g = [], []
    for p in PART.values():
        m = pa == p
        w_r.append(acc_skf(X[m], y[m]))
        w_g.append(acc_grp(X[m], y[m], sgm[m], 5))
    res[f"{key}_within_participant_random_5fold"] = float(np.mean(w_r))
    res[f"{key}_within_participant_grouped_by_segment"] = float(np.mean(w_g))
    res[f"{key}_per_participant_grouped"] = [round(float(v), 1) for v in w_g]
for k, v in res.items():
    print(f"  {k}: {round(v, 2) if not isinstance(v, list) else v}")
json.dump(res, open(os.path.join(OUT, "detected_segments_classification.json"), "w"), indent=2)
