#### This repository provides the software stack for a high-fidelity surface electromyography (sEMG) acquisition system designed for benchmarking low-cost hardware against clinical-grade standards. 
#### The system utilizes an ESP8266 microcontroller and an ADS1015 12-bit ADC (connected to SEN0240 sEMG vía I2C)  to achieve a deterministic sampling rate of 1000 SPS, ensuring strict temporal and spectral alignment with reference databases such as NinaPro DB1.
The firmware is implemented in MicroPython and utilizes an optimized binary data-streaming protocol. This method ensures a stable 1000 Hz sampling frequency by bypassing the Flash memory write latency of the ESP8266. The accompanying offline Python engine performs the complete digital signal processing pipeline, including 4th-order Butterworth band-pass filtering (20–450 Hz), 50 Hz digital notch filtering, full-wave rectification, linear envelope extraction via a 3 Hz leaky integrator, and polyphase resampling to 100 Hz.
#### Hardware integration requires an ESP8266 clocked at 160 MHz connected to an ADS1015 via I2C on GPIO 0 (SCL) and GPIO 2 (SDA). The ADC address must be set to 0x48 by grounding the ADDR pin. The DFRobot SEN0240 sensor is connected to channels A0 and A1 in differential mode.
#### Experimental validation across multiple subjects demonstrated an average Signal-to-Noise Ratio (SNR) over 29 dB and Pearson correlation of 0.80 morphological correlation with clinical-grade signals. This project proves that clinical-level sEMG acquisition requirements can be met using accessible off-the-shelf components.

### Repository Structure:
### -Firmware: MicroPython code for the ESP8266 (acquisition program):
* $${\color{LightBlue} \text{EMG\\_ADS1015\\_1000SPS\\_80s\\_v4.py}}$$   -High-speed binary acquisition script via i2c with ADS1015
### -Processing: Python scripts for PC-based data analysis (to obtain statistics peak, MAV, RMS, WL and visualization):
* $${\color{LightBlue} \text{plotter\\_data(10Rx8seg)\\_binary\\_mv\\_stats\\_2.py}}$$
### -Pearson Correlation Analysis:
* ### -Pearson Correlation Analysis:
* $${\color{LightBlue} \text{Pearson\\_correlation\\_validator\\_with\\_OFFSET\\_SAMPLES.py}}$$
### -Classification: Implements a Random Forest Classifier
* $${\color{LightBlue} \text{sEMG\\_Gesture\\_Recognition.py}}$$
### -SNR Analysis:
* $${\color{LightBlue} \text{sEMG\_SNR\\_calculator.py}}$$

  

## Installation & Usage:
Install MicroPython on your ESP8266. Note: This project was developed using an ESP8266 with 1MB of Flash memory.
1. **Prepare the Hardware:** Install **MicroPython** on your ESP8266.
2. **Upload the Firmware:** Transfer `EMG_ADS1015_1000SPS_80s_v4.py` to the board using **Thonny** or your preferred IDE.
3. **Data Acquisition:** Run the script to start the recording.
   * *Note: The protocol lasts 80 seconds, consisting of 10 repetitions (5s active / 3s rest).*
4. **Data Transfer:** Once finished, download the generated `.bin` file to your PC for processing.

## Offline DSP Processing (PC)
### Offline DSP Processing (PC)
Ensure you have **Python** installed with the following libraries:
* `numpy`
* `matplotlib`
* `scipy`
#### Process and visualize the data (binary):

* * **Run:** `python plotter_data(10Rx8seg)_binary_mv_stats_2.py` &nbsp; `gesture_file.bin`

* This software processes the acquired binary data by applying a **DSP pipeline** to extract key EMG features.
* The results—including **Peak, MAV, RMS, and Waveform Length (WL)**—are calculated and reported via the console.
* Finally, the script generates a **visual plot** of the processed signal for analysis.



  
## DSP Pipeline Details
To match clinical standards, the raw signal undergoes the following stages:
* DC Offset Removal: Centering the signal at 0 mV.
* Band-pass Filter: 4th-order Butterworth (20–450 Hz).
* Notch Filter: Digital IIR at 50 Hz (Q =30).
* Full-wave Rectification: Extracting signal magnitude.
* Linear Envelope: 3 Hz low-pass filter (Leaky Integrator).
* Downsampling: Polyphase resampling to 100 Hz for NinaPro alignment.

### Pearson correlation analysis
#### Configuration
Before running the script, ensure you edit the following fields with the appropriate values inside the code:
* `user_file= = 'fist_gesture_raw_2.bin'`
* `FILE_MAT_NINAPRO = 'S1_A1_E2.mat'`
* `GESTURE_ID = 6`
* `CHANNELS_NINA = 2`
* `OFFSET_SAMPLES = 105`
* **Run:** &nbsp; `python  Pearson_correlation_validator_with_OFFSET_SAMPLES.py`

* This script is designed to estimate the Pearson correlation coefficient for each gesture. 
* The software compares the input binary data directly against the NinaPro DB1 database.
* Requirements:
* The script must be located in the same folder as the binary data file.
* The NinaPro database file (S1_A1_E2) must also be present in the same directory.
* Output:
Visualization: A plot comparing the acquired binary signal vs. the NinaPro reference.(correlation factor included in the plot).
Console Report: Prints the final calculated Pearson correlation factor.

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
Place your all .bin files in the same directory as the script.
Ensure filenames contain the gesture labels: fist, thumb, or spread.
### Execute the script:
code
Bash
* **Run:** &nbsp; `python sEMG_Gesture_Recognition.py`
 
* This software implements a Random Forest classifier using 1,440 sliding windows (200 ms width, 50% overlap).
* It evaluates performance across three distinct gestures and generates a detailed confusion matrix.
* Outputs:
* Confusion matrix for analysis (table 3)
* Step-by-step processing logs (Feature extraction to benchmarking).

### SNR analysis:
* **Run:** &nbsp; `python sEMG_SNR_calculator.py gesture_file.bin`

* The program will detect the noise floor and the peak value in the current session
* The results will be reported by console and also a visulazation of the curve.
### Noise Window Adjustment (Offset)

The script uses a specific **rest window** to measure the hardware base noise (electronic interference) before the muscle is activated.

#### **What is it for?**
This window (set by default from samples **2000 to 3000**) is taken during the rest period (`DUR_REST`). The script calculates the average of this signal to "clean" the background noise and obtain an accurate **SNR (Signal-to-Noise Ratio)**. If your rest signal starts earlier or later, you must move this window to avoid accidentally capturing the beginning of the muscle contraction.

#### **How to change the values?**
Look for the following line inside the `perform_ensemble_analysis` function:

* pythoncode:
## Rest window: from 200 to 300 (100Hz scale = 2000 to 3000 original)
seg_raw_rest = raw_100[start + 200 : start + 300]
* First value (200): The starting point of the noise sample.
* Second value (300): The ending point of the noise sample.
### [!IMPORTANT]
* Scale Note: Since the code processes data resampled at 100Hz, you must divide the original sample value by 10.
* Example: To analyze from sample 1500 to 2500, you should enter [start + 150 : start + 250].
#### **Example Output (Console Report)**
When the script finishes, it will display a technical characterization in the console:

### SNR Report example:
* ======================================================================
 * TECHNICAL CHARACTERIZATION: fist_gesture_raw_2.bin
* ======================================================================
 1. Processed Signal Peak (Asignal):           8.57 mV
 2. Raw Input Noise (Offset 2k-3k):            3.79 mV
 3. Residual Noise (Post-DSP RMS):            0.3119 mV
* ---------------------------------------------------------------------
 * Filtering Efficiency:                         94.18 %
 * Signal-to-Noise Ratio (SNR dB):               28.78 dB
* ======================================================================

### Contac: 
* Author: Fernando Moutinho
* Contact: fmoutinho2012@gamil.com
