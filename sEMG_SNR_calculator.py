import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, resample
import struct
import sys
import os

# --- EXPERIMENTAL CONFIGURATION ---
FS_ORIGINAL = 1000.0   # Original Sampling Frequency (Hz)
FS_TARGET = 100.0      # Target Frequency for plotting (100 Hz)
REPS = 10              # Number of repetitions
DUR_TOTAL = 8.0        # Total trial duration (s)
DUR_REST = 3.0         # Rest duration (s) 
DUR_ACTIVE = 5.0       # Active duration (s) -> 500 samples at 100Hz
NYQ = 0.5 * FS_ORIGINAL

def load_binary_data(file_path):
    """Reads ADS1015 binary data and converts to mV."""
    voltages = []  # Standardized variable name
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(2)
            if len(chunk) < 2: break
            # Extract the integer from the tuple [0]
            raw_val = struct.unpack('>h', chunk)[0] 
            val_12bit = raw_val >> 4
            voltages.append(val_12bit * 2.0) 
    return np.array(voltages)

def apply_full_dsp(signal):
    """Full DSP Chain: Band-pass -> Notch -> Rectify -> 3Hz Low-pass Envelope."""
    # 1. Band-pass (20-450 Hz)
    b, a = butter(4, [20/NYQ, 450/NYQ], btype='band')
    sig_centered = signal - np.mean(signal)
    f1 = filtfilt(b, a, sig_centered)
    
    # 2. Notch (50 Hz)
    bn, an = iirnotch(50, 30, FS_ORIGINAL)
    f2 = filtfilt(bn, an, f1)
    
    # 3. Envelope (Rectification + 3Hz Low-pass)
    rectified = np.abs(f2)
    be, ae = butter(2, 3.0/NYQ, btype='low')
    envelope = filtfilt(be, ae, rectified)
    return envelope

def calculate_noise_transfer(input_noise_amplitude):
    """Simulates noise transfer through the DSP chain to find RMS."""
    t = np.linspace(0, 1, int(FS_ORIGINAL))
    synthetic_noise = input_noise_amplitude * np.sin(2 * np.pi * 50 * t)
    processed_noise = apply_full_dsp(synthetic_noise)
    return np.mean(processed_noise), np.sqrt(np.mean(processed_noise**2))

def perform_ensemble_analysis(raw_data):
    """Segments data, resamples to 100Hz, and averages only the active phase."""
    # 1. Process full signal for envelope
    full_envelope = apply_full_dsp(raw_data)
    
    # 2. Resample both to 100Hz scale
    num_samples_100hz = int(len(full_envelope) * FS_TARGET / FS_ORIGINAL)
    env_100 = resample(full_envelope, num_samples_100hz)
    raw_100 = resample(raw_data, num_samples_100hz)
    
    samples_per_trial = int(DUR_TOTAL * FS_TARGET) # 800 samples
    samples_rest = int(DUR_REST * FS_TARGET)       # 300 samples
    
    active_trials = []
    hardware_offsets = []

    for i in range(REPS):
        start = i * samples_per_trial
        if start + samples_per_trial > len(env_100): break
        
        # Segment Raw Rest for Hardware Noise (rest window: 200-300 in 100Hz scale)
        seg_raw_rest = raw_100[start + 200 : start + 300]
        raw_offset = np.mean(seg_raw_rest) - 1500.0
        hardware_offsets.append(raw_offset)
        
        # Segment Envelope Action (Samples 300 to 800)
        seg_env_rest = env_100[start : start + samples_rest]
        seg_env_active = env_100[start + samples_rest : start + samples_per_trial]
        
        # Peg to Zero: Subtract rest baseline from active phase envelope
        rest_floor = np.mean(seg_env_rest)
        corrected_active = seg_env_active - rest_floor
        active_trials.append(corrected_active)
            
    # Ensemble Average of the 500 samples (Action Phase)
    final_avg_active = np.mean(np.array(active_trials), axis=0)
    final_avg_active[final_avg_active < 0] = 0 
    
    r_in = np.mean(hardware_offsets)
    r_out_mean, r_out_rms = calculate_noise_transfer(np.abs(r_in))
    
    return final_avg_active, r_in, r_out_mean, r_out_rms

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py filename.bin")
        sys.exit()

    file_name = sys.argv[1]
    
    try:
        raw_signal = load_binary_data(file_name)
        # avg_active is exactly 500 samples
        avg_active, r_in, r_out_mean, r_out_rms = perform_ensemble_analysis(raw_signal)
        
        # Dynamic Peak detection on the zero-pegged averaged active signal
        signal_peak = np.max(avg_active)
        
        # SNR Calculation in dB
        snr_db = 20 * np.log10(signal_peak / r_out_rms)
        efficiency = (1 - (r_out_mean / np.abs(r_in))) * 100
        
        # --- TECHNICAL REPORT ---
        print("\n" + "="*70)
        print(f" TECHNICAL CHARACTERIZATION: {os.path.basename(file_name)}")
        print("="*70)
        print(f" 1. Processed Signal Peak (Asignal):     {signal_peak:>10.2f} mV")
        print(f" 2. Raw Input Noise (Offset 2k-3k):      {r_in:>10.2f} mV")
        print(f" 3. Residual Noise (Post-DSP RMS):        {r_out_rms:>10.4f} mV")
        print("-" * 70)
        print(f" Filtering Efficiency:                    {efficiency:>10.2f} %")
        print(f" Signal-to-Noise Ratio (SNR dB):          {snr_db:>10.2f} dB")
        print("="*70 + "\n")
        
        # --- PLOT (X-AXIS 0 TO 500) ---
        plt.figure(figsize=(10, 5))
        plt.plot(avg_active, color='crimson', linewidth=2, label='Averaged Action Phase')
        plt.axhline(y=signal_peak, color='black', linestyle='--', alpha=0.4, label=f'Peak: {signal_peak:.2f} mV')
        
        plt.title(f"Averaged sEMG Action Phase (5s) - {os.path.basename(file_name)}", fontsize=14)
        plt.ylabel("Amplitude (mV)", fontsize=12)
        plt.xlabel("Samples (0 to 500 @ 100Hz)", fontsize=12)
        plt.xlim(0, 500)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"ERROR: {e}")
