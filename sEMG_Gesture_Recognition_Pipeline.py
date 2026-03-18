import numpy as np
import struct
import os
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIGURACIÓN TÉCNICA DEL DATASET (Basado en Paper V18) ---
FS = 1000.0         # Frecuencia de muestreo (Hz)
WINDOW_SIZE = 200    # Ventana de 200ms (200 muestras)
STEP_SIZE = 100      # Desplazamiento de 100ms (50% Overlap)
DURACION_TRIAL = 8.0 # 8 segundos por repetición
SEC_REST = 3.0       # 3 segundos iniciales de reposo (ignorados)
SEC_ACTIVE = 5.0     # 5 segundos de acción (procesados)
REPS = 10            # 10 repeticiones por archivo .bin

def extract_pro_features(w):
    """
    Extrae el set completo de características para el Random Forest.
    Incluye las métricas de amplitud y complejidad temporal.
    """
    mav = np.mean(np.abs(w))
    rms = np.sqrt(np.mean(w**2))
    var = np.var(w)
    wl = np.sum(np.abs(np.diff(w)))
    # Zero Crossings (ZC)
    zc = len(np.where(np.diff(np.sign(w)))[0])
    # Slope Sign Changes (SSC)
    ssc = len(np.where(np.diff(np.sign(np.diff(w))))[0])
    # Willison Amplitude (WAMP) - Umbral de 10mV aproximado tras normalización
    wamp = np.sum(np.abs(np.diff(w)) > 0.01)
    
    return [mav, rms, var, wl, zc, ssc, wamp]

def preprocess_signal(raw_data):
    """Pipeline de procesamiento: Media 0, Band-pass, Notch y Normalización"""
    nyq = 0.5 * FS
    # 1. Filtro Band-pass 20-450 Hz
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    sig = filtfilt(b, a, raw_data - np.mean(raw_data))
    
    # 2. Filtro Notch 50 Hz
    bn, an = iirnotch(50, 30, FS)
    sig = filtfilt(bn, an, sig)
    
    # 3. Normalización Intra-Sujeto (Escalado por el máximo absoluto)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return sig

def load_and_segment(folder_path):
    """Escanea la carpeta, identifica gestos y extrae ventanas activas"""
    X = []
    y = []
    labels_map = {'fist': 0, 'thumb': 1, 'spread': 2}
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".bin")]
    
    for file in files:
        label_id = next((v for k, v in labels_map.items() if k in file.lower()), None)
        if label_id is not None:
            # Leer binario 16-bit
            with open(os.path.join(folder_path, file), "rb") as f:
                raw = []
                while True:
                    chunk = f.read(2)
                    if not chunk: break
                    val = struct.unpack('>h', chunk)[0]
                    raw.append((val >> 4) * 2.0) # Convertir a mV
            
            clean_sig = preprocess_signal(np.array(raw))
            
            # Segmentación de los 10 ciclos
            for rep in range(REPS):
                start_active = int((rep * DURACION_TRIAL + SEC_REST) * FS)
                end_active = int((rep * DURACION_TRIAL + DURACION_TRIAL) * FS)
                segment = clean_sig[start_active:end_active]
                
                # Extracción de ventanas
                for i in range(0, len(segment) - WINDOW_SIZE, STEP_SIZE):
                    window = segment[i : i + WINDOW_SIZE]
                    X.append(extract_pro_features(window))
                    y.append(label_id)
                    
    return np.array(X), np.array(y)

# --- PROGRAMA PRINCIPAL ---
if __name__ == "__main__":
    current_dir = "."
    print("\n[STEP 1] Scanning folder and extracting features...")
    
    X, y = load_and_segment(current_dir)
    
    if len(X) == 0:
        print("ERROR: No .bin files found with 'fist', 'thumb', or 'spread' in their names.")
        exit()

    # --- AUDITORÍA DE DATOS ---
    print("="*60)
    print(f" DATASET AUDIT: {len(X)} feature vectors (windows) successfully extracted.")
    print(f" Classes detected: {np.unique(y)}")
    print("="*60)

    # --- VALIDACIÓN 1: 3 GESTOS (RANDOM FOREST + K-FOLD) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_3 = []
    cm_3 = np.zeros((3, 3))

    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs_3.append(accuracy_score(y[test_idx], preds))
        cm_3 += confusion_matrix(y[test_idx], preds)

    # --- VALIDACIÓN 2: BINARIA (FIST VS THUMB) ---
    mask = (y == 0) | (y == 1)
    X_bin, y_bin = X[mask], y[mask]
    skf_bin = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_2 = []

    for train_idx, test_idx in skf_bin.split(X_bin, y_bin):
        clf_bin = RandomForestClassifier(n_estimators=200, random_state=42)
        clf_bin.fit(X_bin[train_idx], y_bin[train_idx])
        accs_2.append(accuracy_score(y_bin[test_idx], clf_bin.predict(X_bin[test_idx])))

    # --- RESULTADOS FINALES PARA EL PAPER ---
    print("\n[STEP 2] Benchmark Results:")
    print("-" * 60)
    print(f" ACCURACY (Full 3-Gesture Set):  {np.mean(accs_3)*100:.2f}%")
    print(f" ACCURACY (Binary: Fist vs Thumb): {np.mean(accs_2)*100:.2f}%")
    print("-" * 60)
    
    print("\n[STEP 3] Confusion Matrix (3-Gesture Set) for Table 3:")
    print(cm_3.astype(int))
    print("\n[STEP 4] Per-Gesture Recall:")
    gestures = ['Fist', 'Thumb Up', 'Finger Spread']
    for i, g in enumerate(gestures):
        recall = cm_3[i,i] / np.sum(cm_3[i,:])
        print(f" {g}: {recall*100:.2f}%")
    
    print("\n" + "="*60)
    print(" Analysis complete. Data is ready for Paper V18 and GitHub.")
    print("="*60 + "\n")