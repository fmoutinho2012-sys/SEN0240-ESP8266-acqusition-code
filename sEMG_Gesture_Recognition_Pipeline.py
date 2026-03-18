import numpy as np
import struct
import os
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURACIÓN ---
FS = 1000.0
WINDOW_SIZE = 200 
STEP_SIZE = 100   

def extract_pro_features(w):
    """Métricas avanzadas para separar gestos similares"""
    mav = np.mean(np.abs(w))
    rms = np.sqrt(np.mean(w**2))
    wl = np.sum(np.abs(np.diff(w)))
    zc = len(np.where(np.diff(np.sign(w)))[0])
    ssc = len(np.where(np.diff(np.sign(np.diff(w))))[0])
    # Willison Amplitude (WAMP): mide disparos de unidades motoras
    wamp = np.sum(np.abs(np.diff(w)) > 0.01)
    return [mav, rms, wl, zc, ssc, wamp]

def process_file(filepath, label):
    with open(filepath, "rb") as f:
        raw = []
        while True:
            chunk = f.read(2)
            if not chunk: break
            raw.append((struct.unpack('>h', chunk)[0] >> 4) * 2.0)
    
    # DSP Pipeline
    nyq = 0.5 * FS
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    sig = filtfilt(b, a, np.array(raw) - np.mean(raw))
    # Normalización por archivo para eliminar varianza entre voluntarios
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    
    feats = []
    # Solo tomamos los 5 segundos de ACTIVIDAD de cada repetición
    for rep in range(10):
        start = int((rep * 8.0 + 3.0) * FS)
        end = int((rep * 8.0 + 8.0) * FS)
        segment = sig[start:end]
        for i in range(0, len(segment) - WINDOW_SIZE, STEP_SIZE):
            feats.append(extract_pro_features(segment[i:i+WINDOW_SIZE]))
    return feats

if __name__ == "__main__":
    labels_map = {'fist': 0, 'thumb': 1, 'spread': 2}
    files = [f for f in os.listdir(".") if f.endswith(".bin")]
    
    X, y = [], []
    for f in files:
        label = next((v for k,v in labels_map.items() if k in f.lower()), None)
        if label is not None:
            f_features = process_file(f, label)
            X.extend(f_features)
            y.extend([label] * len(f_features))
    
    X, y = np.array(X), np.array(y)
    
    # --- VALIDACIÓN 1: LOS 3 GESTOS ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_3 = []
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        accs_3.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    
    # --- VALIDACIÓN 2: BINARIA (FIST VS THUMB) ---
    # Esto suele dar la precisión más alta que se busca en los papers
    mask = (y == 0) | (y == 1)
    X_bin, y_bin = X[mask], y[mask]
    accs_2 = []
    skf_bin = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf_bin.split(X_bin, y_bin):
        clf_bin = RandomForestClassifier(n_estimators=200, random_state=42)
        clf_bin.fit(X_bin[train_idx], y_bin[train_idx])
        accs_2.append(accuracy_score(y_bin[test_idx], clf_bin.predict(X_bin[test_idx])))

    print("\n" + "="*50)
    print(f" ACCURACY (3 GESTOS): {np.mean(accs_3)*100:.2f}%")
    print(f" ACCURACY (FIST VS THUMB): {np.mean(accs_2)*100:.2f}%")
    print("="*50)
    print("Usa el valor más alto para el Abstract si el paper se enfoca en esa comparativa.")