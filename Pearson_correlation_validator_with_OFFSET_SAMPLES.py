import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, resample
import struct
import os

# =============================================================
# 1. CONFIGURACIÓN Y ALINEACIÓN
# =============================================================

# --- CAMBIA AQUÍ EL NOMBRE DE TU ARCHIVO .BIN ---
ARCHIVO_BIN_USUARIO = 'fingers_spread_gesture_raw_3.bin' 

# --- AJUSTE DE ALINEACIÓN (OFFSET) ---
# Si tu señal empieza antes que NinaPro, usa un valor POSITIVO.
# Prueba con valores entre 50 y 100 según tu gráfica.
OFFSET_SAMPLES = -17# <--- AJUSTA ESTE VALOR PARA MOVER TU CURVA
# -------------------------------------

FILE_MAT_NINAPRO = 'S1_A1_E2.mat' 
GESTURE_ID = 5
CHANNELS_NINA = 1

# Parámetros técnicos
FS_TARGET = 100.0     
DURACION_ACTIVE = 5.0  
REPS = 10              

# =============================================================
# 2. FUNCIONES (Mantienen la lógica anterior)
# =============================================================

def cargar_binario_ads1015(fichero):
    voltajes = []
    with open(fichero, "rb") as f:
        while True:
            chunk = f.read(2)
            if len(chunk) < 2: break
            raw = struct.unpack('>h', chunk)[0]
            val_12bit = raw >> 4
            voltajes.append(val_12bit * 2.0) 
    return np.array(voltajes)

def dsp_pipeline(data):
    # (Procesamiento idéntico al anterior para mantener coherencia)
    fs_original = 1000.0
    nyq = 0.5 * fs_original
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    sig = filtfilt(b, a, data - np.mean(data))
    bn, an = iirnotch(50, 30, fs_original)
    sig = filtfilt(bn, an, sig)
    rectified = np.abs(sig)
    be, ae = butter(2, 3.0/nyq, btype='low')
    envelope = filtfilt(be, ae, rectified)
    sig_100 = resample(envelope, int(len(envelope) * FS_TARGET / fs_original))
    
    s_per_trial = int(8.0 * FS_TARGET)
    s_rest = int(3.0 * FS_TARGET)
    active_trials = []
    for i in range(REPS):
        idx_ini = i * s_per_trial
        reposo = sig_100[idx_ini : idx_ini + s_rest]
        accion = sig_100[idx_ini + s_rest : idx_ini + s_per_trial]
        if len(accion) == int(DURACION_ACTIVE * FS_TARGET):
            active_trials.append(accion - np.mean(reposo))
    avg = np.mean(np.array(active_trials), axis=0)
    avg[avg < 0] = 0 
    return avg

# =============================================================
# 3. EJECUCIÓN CON DESPLAZAMIENTO
# =============================================================

try:
    # 1. Cargar NinaPro
    mat = scipy.io.loadmat(FILE_MAT_NINAPRO)
    emg_nina_full = mat['emg']
    stimulus = mat['stimulus'].flatten()
    idx_nina = np.where(stimulus == GESTURE_ID)[0]
    curva_nina = emg_nina_full[idx_nina[0]:idx_nina[0]+500, CHANNELS_NINA-1]

    # 2. Cargar y procesar tu señal
    curva_bin_base = dsp_pipeline(cargar_binario_ads1015(ARCHIVO_BIN_USUARIO))

    # --- APLICAR OFFSET ---
    if OFFSET_SAMPLES > 0:
        # Añade ceros al inicio y recorta el final
        curva_bin = np.pad(curva_bin_base, (OFFSET_SAMPLES, 0), mode='constant')[:500]
    elif OFFSET_SAMPLES < 0:
        # Recorta el inicio y añade ceros al final
        curva_bin = np.pad(curva_bin_base[abs(OFFSET_SAMPLES):], (0, abs(OFFSET_SAMPLES)), mode='constant')
    else:
        curva_bin = curva_bin_base

    # 3. Correlación
    correlacion = np.corrcoef(curva_nina, curva_bin)[0, 1]

    # 4. Plot
    plt.figure(figsize=(12, 5))
    plt.plot(curva_nina / np.max(curva_nina), label='NinaPro (Ref)', color='gray', alpha=0.5)
    plt.plot(curva_bin / np.max(curva_bin), label=f'Our (Offset: {OFFSET_SAMPLES})', color='royalblue', linewidth=2)
    
    plt.title(f'Pearson Correlation with Time Lag Adjustment | Pearson: {correlacion:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

    print(f"Nueva Correlación con Offset {OFFSET_SAMPLES}: {correlacion:.4f}")

except Exception as e:
    print(f"Error: {e}")
