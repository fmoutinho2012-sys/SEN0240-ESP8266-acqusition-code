import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, resample
import struct
import sys
import os

# --- CONFIGURACIÓN DEL EXPERIMENTO (Basado en NinaPro DB1) ---
FS_ORIGINAL = 1000.0  # Muestreo del ADS1015
FS_TARGET = 100.0     # Muestreo objetivo (NinaPro)
DURACION_TOTAL_TRIAL = 8.0 # 3s Reposo + 5s Acción
DURACION_REST = 3.0        
DURACION_ACTIVE = 5.0      
REPS = 10

def cargar_binario_ads1015(fichero):
    """Lee el binario del ESP8266 y lo convierte a mV"""
    voltajes = []
    if not os.path.exists(fichero):
        raise FileNotFoundError(f"No se encuentra el archivo: {fichero}")
        
    with open(fichero, "rb") as f:
        while True:
            chunk = f.read(2)
            if len(chunk) < 2: break
            # Desempaquetar Big-Endian ('>h')
            raw = struct.unpack('>h', chunk)[0]
            # El ADS1015 en 12-bit shiftado a la izquierda
            val_12bit = raw >> 4
            # Conversión a mV (Asumiendo ganancia x2 del ADS1015 -> 2.048V / 2048)
            voltajes.append(val_12bit * 2.0) 
    return np.array(voltajes)

def dsp_pipeline_nina_style(data):
    """Procesamiento digital de señales siguiendo el estándar clínico"""
    nyq = 0.5 * FS_ORIGINAL
    
    # 1. Filtro Band-pass (20-450 Hz) - 4º Orden
    b, a = butter(4, [20/nyq, 450/nyq], btype='band')
    sig = filtfilt(b, a, data - np.mean(data))
    
    # 2. Filtro Notch (50 Hz) para interferencia eléctrica
    bn, an = iirnotch(50, 30, FS_ORIGINAL)
    sig = filtfilt(bn, an, sig)
    
    # 3. Rectificación (Full-wave rectification)
    rectified = np.abs(sig)
    
    # 4. Envolvente Lineal (Pasa-bajo 3Hz)
    be, ae = butter(2, 3.0/nyq, btype='low')
    envelope = filtfilt(be, ae, rectified)
    
    # 5. Resample a 100Hz para alineación con base de datos clínica
    num_muestras_100hz = int(len(envelope) * FS_TARGET / FS_ORIGINAL)
    sig_100 = resample(envelope, num_muestras_100hz)
    
    # 6. Segmentación y Corrección de Baseline
    samples_per_trial = int(DURACION_TOTAL_TRIAL * FS_TARGET) # 800 pts
    samples_rest = int(DURACION_REST * FS_TARGET)           # 300 pts
    
    active_trials = []
    for i in range(REPS):
        idx_inicio = i * samples_per_trial
        reposo = sig_100[idx_inicio : idx_inicio + samples_rest]
        accion = sig_100[idx_inicio + samples_rest : idx_inicio + samples_per_trial]
        
        if len(accion) == int(DURACION_ACTIVE * FS_TARGET):
            # Baseline Subtraction (Eliminamos el ruido medido en el reposo)
            noise_floor = np.mean(reposo)
            corrected_trial = accion - noise_floor
            active_trials.append(corrected_trial)
    
    # Promedio de las 10 repeticiones para fidelidad morfológica
    avg_signal = np.mean(np.array(active_trials), axis=0)
    avg_signal[avg_signal < 0] = 0 # Asegurar positividad física
    
    return avg_signal

def calculate_metrics(signal):
    """Calcula las métricas de tiempo real para la Tabla 2"""
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    wl = np.sum(np.abs(np.diff(signal)))
    max_amp = np.max(signal)
    return mav, rms, wl, max_amp

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python semg_analyzer_v15.py nombre_del_archivo.bin")
        sys.exit()

    archivo = sys.argv[1]
    
    try:
        raw_voltages = cargar_binario_ads1015(archivo)
        processed_envelope = dsp_pipeline_nina_style(raw_voltages)
        
        # Obtener métricas reales
        mav, rms, wl, max_amp = calculate_metrics(processed_envelope)
        
        # Resultados por consola formateados para el Paper
        print("\n" + "="*50)
        print(f" RESULTADOS TÉCNICOS: {archivo}")
        print("="*50)
        print(f" Signal Peak (Max):  {max_amp:>10.2f} mV")
        print(f" MAV (Mean Abs):     {mav:>10.2f} mV")
        print(f" RMS:                {rms:>10.2f} mV")
        print(f" Waveform Length:    {wl:>10.2f}")
        print("-" * 50)
        print(" INFO: Datos listos para Tabla 2 de la Versión V15")
        print("="*50 + "\n")
        
        # Plot visual
        plt.figure(figsize=(10, 5))
        plt.plot(processed_envelope, color='royalblue', linewidth=2.5, label=f'Processed {archivo}')
        plt.title(f"Averaged sEMG Envelope (10 Reps) - {archivo}", fontsize=14)
        plt.xlabel("Samples at 100 Hz (Time Alignment)", fontsize=12)
        plt.ylabel("Amplitude (mV)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"ERROR: {e}")