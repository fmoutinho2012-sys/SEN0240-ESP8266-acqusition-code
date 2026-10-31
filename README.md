# Single-channel low-cost sEMG acquisition chain: SEN0240 + ADS1015 + ESP8266

Open firmware, raw recordings and analysis code for a single-channel surface EMG (sEMG) chain built from commodity modules. The repository accompanies a paper that characterises the chain at signal level and shows how validation design affects classification accuracy.

**Scope.** This is a research prototype and not a medical device. No leakage-current, isolation or other IEC 60601 tests were performed, and the repository makes no claim of clinical-grade performance. Gesture decoding is **not** demonstrated by these data (see "What the data do and do not show").

## Hardware used for the recordings

| Part | Role | Notes |
|---|---|---|
| DFRobot SEN0240 (OYMotion) | Active sEMG sensor, dry electrodes | ±1.5 mV input, gain 1000, 0–3.0 V output around 1.5 V, 20–500 Hz effective spectrum (vendor data) |
| ADS1015 breakout | 12-bit ΔΣ converter | I2C address 0x48 |
| IoT Deployment Board v3.3 | ESP8266 board designed by the author (ESP-01 module, AMS1117 3.3 V regulator), MicroPython | 160 MHz; schematic and Gerber files in `hardware/iot_deployment_board_v3.3/` |

The board was powered from a pack of four Ni-Cd cells (4.8 V measured); the ADS1015 and the sensor were supplied at 3.3 V (the converter from the board's regulator, the sensor from the converter module's supply pins); all ground connections were joined at a single node. Data were stored in the board's flash and transferred afterwards; the USB cable to the computer (a laptop on its own battery, charger disconnected) was connected during the recordings. The cost is about €65 (sensor €42.50, ADS1015 module €2.50, board about US$22); see `hardware/BOM.md`.

**ADS1015 configuration (register 0x01 = 0x4283).** Single-ended, AIN0 versus GND (MUX = 100b); full scale ±4.096 V, 2 mV per code (2 µV referred to the sensor input); continuous conversion at 1600 SPS. The sensor output (0–3.0 V) therefore uses at most about 1500 codes (about 10.6 bits). The ESP8266 reads the converter at 1000 Hz, paced by its microsecond timer and without the converter's ready signal, so each sample is the latest completed conversion and its time stamp is uncertain by up to 0.625 ms. In addition, the firmware writes the data to flash in blocks of 100 samples (100 ms); the read loop falls behind schedule during each write and then catches up, so the read intervals are irregular after every block boundary (see the 250 Hz test below). The recordings carry no time stamps. The sensor is **not** connected in differential mode.

Wiring: I2C on GPIO 0 (SCL) and GPIO 2 (SDA), 400 kHz; ADDR pin to ground.

## Repository layout

| Folder | Content |
|---|---|
| `firmware/` | MicroPython acquisition script `EMG_ADS1015_1000SPS_80s_v4.py` and the MicroPython image used |
| `data/` | 18 raw recordings (6 participants × 3 gestures) and the participant table. File index 7 is participant S6. `data/tests/Test_{1,50,100,250,450}Hz_raw.bin` are the digitised test tones (80 s each, same firmware; the 1 Hz file is a sine of 1.5 Vpp with +0.75 V offset, the others 1.0 Vpp with +1.5 V offset) |
| `analysis/` | `reproduce_analysis.py` regenerates every number of Section 3 of the paper; `reanalysis_detected_segments.py` repeats the classification with windows from detected contraction and rest segments and estimates the burst period; `analyse_250Hz_test.py` and `analyse_tone_tests.py` analyse the 250 Hz tone and the four tones; `analyse_1Hz_test.py` analyses the 1 Hz sine (fit, linearity, timing of the reads by position in the block); `simulate_timing_effect.py` simulates the effect of the inferred read timing on the features |
| `legacy/` | Earlier scripts kept for traceability. Their outputs are **not** comparable with the paper (see below) |
| `hardware/` | Schematic and Gerber files of the board used; bill of materials; `bench_checks/` oscilloscope captures of the 250 Hz test (time domain and FFT) |

## Recording format

80 s files of 80,000 samples, 16-bit big-endian words from the ADS1015 (12 bits, left-justified; use `word >> 4`). Nominal protocol: ten 8 s cycles of 5 s contraction and 3 s rest, cued verbally by the author ("contraction", "rest") and timed with a stopwatch; the cues were not recorded with the signal. The recorded bursts repeat about every 4.3 s (median; range 2.0–6.6 s), so the nominal timing does not describe the data. Each participant made one recording per gesture (fist, thumb up, finger spread) with the sensor repositioned for each gesture.

## Reproducing the results

```bash
pip install numpy scipy pandas scikit-learn
python analysis/reproduce_analysis.py data results
python analysis/reanalysis_detected_segments.py data results
python analysis/analyse_250Hz_test.py data/tests/Test_250Hz_raw.bin results
python analysis/analyse_tone_tests.py data/tests results
python analysis/analyse_1Hz_test.py data/tests/Test_1Hz_raw.bin results
python analysis/simulate_timing_effect.py data data/tests/Test_100Hz_raw.bin results 20
```

Outputs `results/per_recording.csv`, `results/classification.json`, `results/period_per_recording.csv` and `results/detected_segments_classification.json` and prints:

- per-recording noise floor (RMS in rest segments, referred to the sensor input) and detection-based SNR;
- the burst period of each recording;
- the Random Forest accuracy under several validation designs, on nominal-grid windows and on windows from detected segments, with a rest-window negative control.

Processing: zero-phase 4th-order Butterworth band-pass 20–450 Hz, 50 Hz notch (Q = 30), rectification and a 2nd-order zero-phase 3 Hz low-pass for the envelope. Features: MAV, RMS, VAR, WL, ZC, SSC, WAMP over 200 ms windows with 100 ms steps; Random Forest with 200 trees.

## What the data do and do not show

- Rest noise floor referred to the sensor input: median 69 µV RMS (range 52–281 µV) in 20–450 Hz.
- Detection-based SNR: mean 6.3 dB (range 3.1–12.8 dB). This is an upper estimate, because the segments are chosen from the signal itself.
- Digitised test tones (1.0 Vpp on the oscilloscope at the converter input; 50, 100, 250 and 450 Hz): the tone line holds only 26, 24, 20 and 17 % of the power (coherent SNR −4.6, −5.1, −6.1 and −7.0 dB), against 25, 19, 11 and 6 dB expected from the conversion timing alone, with a 10 Hz comb (the 100 ms block period). In 23–25 of 80 one-second windows the strongest line lies at about 0.73 of the tone frequency, that is, samples taken about 0.73 ms apart; the phase of the 50 and 100 Hz tones gives read intervals of about 0.7 ms in the first 40 samples of each 100-sample block and about 0.95 ms in the rest. Because the mean rate is exactly 1000 reads per second, the pause at each block boundary is inferred at about 11–14 ms (not measured). In the steady part of the block (positions 60–95) the timing scatter per read is 184, 193, 197 and 211 µs at 50, 100, 250 and 450 Hz (about 180 µs expected from the 0.625 ms conversion window). A simulation with this inferred timing (stationary signal with the mean spectrum of the recordings, 20 realisations) lowers ZC and SSC by about 8 %, WL by 9 % and MNF and MDF by about 4 % (window scatter up to 10 % and 22 %), and leaves MAV, RMS and VAR within their spread between realisations; it is a simulation, not a measurement on the recordings.
- Digitised 1 Hz sine (generator: 1.5 Vpp, +0.75 V offset, i.e. 750 mV offset and 750 mV amplitude): 80 000 samples, codes 18–754, no saturation. Sine fit to the whole record: offset 765.7 mV, amplitude 725.5 mV, frequency 0.99995 Hz (residual 65.5 mV rms because the block timing is irregular). Fit to the steady part of each block (positions 60–99, free time shift per block): offset 768.8 mV, amplitude 729.7 mV (−2.7 % against 750 mV), residual 1.5 mV rms in steady blocks, no level-dependent error above 0.5 mV. The amplitude and offset differences from the set values are not explained (generator, converter gain and offset and source loading were not checked separately). Timing: the samples at the start of each block are about 9 ms late and recover over about 35 reads (about 10 ms of extra time at the block boundary, quartiles 6.5–10.9 ms); about every 2 s (20.4 blocks, close to 4096 bytes of flash) a longer stall of 53–80 ms follows, a cause that was not tested.
- Contractions repeat about every 4.3 s, not every 8 s, so windows from the nominal 8 s grid mix rest and contraction.
- A Random Forest reaches 81.9 % on three gestures in the original three-file design on nominal windows. With windows from detected segments, true-rest windows are classified better than contraction windows within participants (79.7 % against 69.8 %, folds grouped by segment), and with leave-one-participant-out the accuracy on contraction windows (47.4 %) is close to the majority-class baseline (43.8 %). The classifier identifies the recording, not the gesture.
- Bench checks (shorted sensor input, converter alone, 50 Hz common-mode injection), AC-coupled at 5 mV/div on the converter input pins, gave residuals below 1 mV peak to peak, which is below the converter step; these are upper bounds, not noise or CMRR measurements.
- Electrical measurements on the complete system (board, ADS1015 and sensor; bench supply at 5.0 V, Wi-Fi radio active; the recording firmware switches the radio off): idle current 74.5 mA; peak current in radio transmission 280 mA (1 Ω shunt, single shot); rise time of the 3.3 V rail 3.8 ms, with a 3.38 V maximum overshoot before settling at 3.3 V; I2C clock 400 kHz; read rate 1000 per second (a GPIO toggled at each read gives 500 Hz) (a persistence capture of the toggle edge showed 42 µs peak to peak; it is not used, because it is incompatible with the digitised tones, which are the basis of the timing inferred above). Supply ripple at the 3.3 V rail (regulator output, Wi-Fi active) was 18.5 mVpp (5 mV/div, AC-coupled; criterion 50 mVpp), mainly at 125 kHz, attributed to the bench supply. The 1 Hz and 250 Hz injected signals and the I2C waveforms were checked on an oscilloscope as pass/fail observations without stored values. These are single readings, not an IEC 60601 test. The current drawn during the recordings (Ni-Cd pack, radio off) was not measured.

## Legacy scripts

The scripts in `legacy/` were written for an earlier version of the work.

- `sEMG_SNR_calculator.py` reports peak amplitude of an averaged envelope divided by the residual RMS after filtering (the source of earlier SNR figures above 28 dB). It is not the SNR definition used in the paper and its values must not be compared with it.
- `sEMG_Gesture_Recognition.py` is the original classification script. Its accuracy depends on window overlap and on random folds; `analysis/reproduce_analysis.py` reports it under six designs.
- `Pearson_correlation_validator_with_OFFSET_SAMPLES.py` correlates the envelope with NinaPro DB1 using a fixed offset and averaging over the nominal 8 s cycles. Because the recorded bursts do not follow the 8 s grid, its values are not used in the paper.
- `plotter_data(10Rx8seg)_binary_mv_stats_2.py` plots and prints amplitude statistics. Some comments call the processing "clinical"; it is not.

## Licence and citation

[AUTHOR: choose a licence for code (for example MIT) and for data (for example CC BY 4.0), and add a CITATION file once the paper has a reference.]
