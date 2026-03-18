#### This repository provides the software stack for a high-fidelity surface electromyography (sEMG) acquisition system designed for benchmarking low-cost hardware against clinical-grade standards. 
#### The system utilizes an ESP8266 microcontroller and an ADS1015 12-bit ADC to achieve a deterministic sampling rate of 1000 SPS, ensuring strict temporal and spectral alignment with reference databases such as NinaPro DB1.
The firmware is implemented in MicroPython and utilizes an optimized binary data-streaming protocol. This method ensures a stable 1000 Hz sampling frequency by bypassing the Flash memory write latency of the ESP8266. The accompanying offline Python engine performs the complete digital signal processing pipeline, including 4th-order Butterworth band-pass filtering (20–450 Hz), 50 Hz digital notch filtering, full-wave rectification, linear envelope extraction via a 3 Hz leaky integrator, and polyphase resampling to 100 Hz.
#### Hardware integration requires an ESP8266 clocked at 160 MHz connected to an ADS1015 via I2C on GPIO 0 (SCL) and GPIO 2 (SDA). The ADC address must be set to 0x48 by grounding the ADDR pin. The DFRobot SEN0240 sensor is connected to channels A0 and A1 in differential mode.
#### Experimental validation across multiple subjects demonstrated a Signal-to-Noise Ratio (SNR) of 51.3 dB and a Pearson correlation of 0.91 with clinical-grade signals. This project proves that clinical-level sEMG acquisition requirements can be met using accessible off-the-shelf components.

### Repository Structure:
### -Firmware: MicroPython code for the ESP8266 (acquisition program):
$${\color{LightBlue} \text{EMG\_ADS1015\_1000SPS\_80s\_v4.py}}$$  -High-speed binary acquisition script via i2c with ADS1015
### -Processing: Python scripts for PC-based data analysis (to obtain statistics peak, MAV, RMS, WL and visualization):
plotter_data(10Rx8seg)_binary_mv_stats_2.py
### -Classification: Implements a Random Forest Classifier
sEMG_Gesture_Recognition_Pipeline.py 

## Installation & Usage:
Install MicroPython on your ESP8266.
  * Upload EMG_ADS1015_1000SPS_80s.py using Thonny or your preferred Python IDE.
  * Run the script. It will record 80 seconds of raw binary data (10 reps of 5s active / 3s rest).
  * Download the resulting .bin file to your PC.
## Processing (PC):
Ensure you have Python installed with the following libraries:
code
Bash
* pip install numpy scipy matplotlib
Process and visualize the data (binary):
* run: plotter_data(10Rx8seg)_binary_mv_stats_2.py  &nbsp; your_data_file.bin
  

## DSP Pipeline Details
To match clinical standards, the raw signal undergoes the following stages:
* DC Offset Removal: Centering the signal at 0 mV.
* Band-pass Filter: 4th-order Butterworth (20–450 Hz).
* Notch Filter: Digital IIR at 50 Hz (Q =30).
* Full-wave Rectification: Extracting signal magnitude.
* Linear Envelope: 3 Hz low-pass filter (Leaky Integrator).
* Downsampling: Polyphase resampling to 100 Hz for NinaPro alignment.

### Classification: Implements a Random Forest Classifier with 5-fold Stratified Cross-Validation to benchmark system accuracy for both 3-gesture sets and binary command sets.
### Requirements
To run the analysis, you need Python 3.8+ and the following libraries:
* numpy: For numerical processing and vector operations.
* scipy: For signal processing (filters and resampling).
* scikit-learn: For the Random Forest model and statistical validation metrics.
* matplotlib: (Optional) For signal visualization.

 ##Install dependencies via pip:
code
Bash
* pip install numpy scipy scikit-learn

## How to Use
Place your .bin files in the same directory as the script.
Ensure filenames contain the gesture labels: fist, thumb, or spread.
### Execute the script:
code
Bash
* python sEMG_Gesture_Recognition_Pipeline.py 


### Contac: 
* Author: Fernando Moutinho
* Contact: fmoutinho2012@gamil.com
