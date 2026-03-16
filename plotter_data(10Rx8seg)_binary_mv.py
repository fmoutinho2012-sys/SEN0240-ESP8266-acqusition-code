import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, resample
import struct
import sys
import os

# --- CONFIGURACIÓN DEL EXPERIMENTO (3s Reposo + 5s Acción) ---
FS_ORIGINAL = 1000.0  
FS_TARGET = 100.0     
DURACION_TOTAL_TRIAL = 8.0 # Segundos totales (3+5)
DURACION_REST = 3.0        # Segundos de reposo inicial
DURACION_ACTIVE = 5.0      # Segundos de contracción
REPS = 10

def cargar_binario_ads1015(fichero):
    voltajes = []
    with open(fichero, "rb") as f:
        while True:
            chunk = f.read(2)
            if len(chunk) < 2: break
            raw = struct.unpack('>h', chunk)[0]
            val_12bit = raw >> 4
            voltajes.append(val_12bit * 2.0) # Convertir a mV
    return np.array(voltajes)

def dsp_pipeline_nina_style(data):
    nyq = 0.5 * FS_ORIGINAL
    
    # 1. Filtros de limpieza (Bipolar)
    # Band-pass 20-450 Hz
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    sig = filtfilt(b, a, data - np.mean(data))
    
    # Notch 50 Hz
    bn, an = iirnotch(50, 30, FS_ORIGINAL)
    sig = filtfilt(bn, an, sig)
    
    # 2. Rectificación y Envolvente (Unipolar)
    rectified = np.abs(sig)
    # Filtro pasa-bajo de 3Hz para suavizar la curva (Integrador)
    be, ae = butter(2, 3.0/nyq, btype='low')
    envelope = filtfilt(be, ae, rectified)
    
    # 3. Resample a 100Hz (Alineación temporal)
    num_muestras_100hz = int(len(envelope) * FS_TARGET / FS_ORIGINAL)
    sig_100 = resample(envelope, num_muestras_100hz)
    
    # 4. SEGMENTACIÓN Y CORRECCIÓN DE BASELINE
    samples_per_trial = int(DURACION_TOTAL_TRIAL * FS_TARGET) # 800 pts
    samples_rest = int(DURACION_REST * FS_TARGET)           # 300 pts
    samples_active = int(DURACION_ACTIVE * FS_TARGET)       # 500 pts
    
    active_trials = []
    
    for i in range(REPS):
        inicio_bloque = i * samples_per_trial
        # El reposo son los primeros 3 segundos del bloque
        reposo = sig_100[inicio_bloque : inicio_bloque + samples_rest]
        # La acción son los 5 segundos siguientes
        accion = sig_100[inicio_bloque + samples_rest : inicio_bloque + samples_per_trial]
        
        if len(accion) == samples_active:
            # RESTAMOS el promedio del reposo (Baseline Subtraction)
            # Esto hace que si el músculo está relajado, la gráfica marque 0
            nivel_ruido_base = np.mean(reposo)
            active_trials.append(accion - nivel_ruido_base)
    
    # Promediamos las 10 repeticiones
    avg_signal = np.mean(np.array(active_trials), axis=0)
    
    # Aseguramos que no haya valores negativos por el promedio
    avg_signal[avg_signal < 0] = 0
    
    return avg_signal

# --- EJECUCIÓN ---
if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else "fist_gesture_raw_1.bin"
    
    try:
        raw_data = cargar_binario_ads1015(archivo)
        final_envelope = dsp_pipeline_nina_style(raw_data)
        
        # Crear la gráfica con estilo clínico
        plt.figure(figsize=(10, 5))
        plt.plot(final_envelope, color='blue', linewidth=2, label='SEN0240 (Processed)')
        
        plt.title(f"EMG Signal: {archivo} (Thumb Up Gesture - Averaged 10 Reps)", fontsize=12)
        plt.xlabel("Samples (at 100Hz)", fontsize=10)
        plt.ylabel("Amplitude (mV)", fontsize=10)
        
        # Ajustamos el eje Y para que empiece en 0 como NinaPro
        plt.ylim(0, np.max(final_envelope) * 1.2)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("Procesamiento completado. La señal ahora está alineada con NinaPro.")
        
    except Exception as e:
        print(f"Error: {e}")