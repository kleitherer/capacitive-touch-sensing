"""
Write (and submit) the appropriate Python code to drive out a PRBS signal on 5 GPIO lines
coming out of the RPi4. 

The PRBS length should be flexible, since you will be experimenting
with different lengths (up to a 16-bit shift register, or a sequence length of 65535). 

All 5 drive lines should use the same PRBS code, with each drive using a different phase of the code
(this ensures orthogonality of the signals from each drive line). Since the ADC is clocked at
30kHz, the PRBS signal should be clocked at 15kHz to not require timing recovery. 

Keep in mind, the drive clock and the ADC sense clock are not phase locked in this system, so there
will be timing drift. Depending on how well you optimize your code the actual sampling rate of
the ADC may be lower (>20 kHz). You can also try to clock the PRBS at the same rate as
the ADC if you find the timing drift is small.
"""

import csv
import matplotlib.pyplot as plt
import time
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256
import spidev
import RPi.GPIO as GPIO
import numpy as np
import PRBS_code_SG as prbs



# we're setting the drive lines up ourselves ... so we have to use readADCdata
ADC = ADS1256.ADS1256()
ADC.ADS1256_init()
# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21, GPIO.OUT) # drive 5 (pin 4)
GPIO.setup(20, GPIO.OUT) # drive 4 (pin 3)
GPIO.setup(16, GPIO.OUT) # drive 3 (pin 2)
GPIO.setup(12, GPIO.OUT) # drive 2 (pin 1)
GPIO.setup(7, GPIO.OUT) # drive 1 (pin 0)




# we need to iterate through each channel...
# do we do getChannalValue or readADCvalue

#adc_value = ADC.ADS1256_GetChannalValue(7)*5.0/0x7fffff # this should be clocked at 30kHz -- this is only ch 7 



# we're driving the 5 GPIO pins and each PRBS signal is associated with a delay since each drive line has its own delay
# it comes from the GPIO pins ... do we need to initialize them? each one
#prbs_signal = bpsk(lfsr(...))
#GPIO.output(21, prbs_signal)

#touch_signal = autocorrelation(prbs_signal,adc_value,511)
#milo giving us a hint: 
#can only read one channel at a time, pick one channel and drive your all 5 drive lines entirety of the sequence, read it, correlate it, then move on to the next sense line and repeat. 
# compare correlations to baseline of what we read when no touch is present. 
# we jsut wnt ot know where it is along the channel by seeing where and how large the peaks are 
# for each channel we do the correlation once . 


# we're driving the prbs signal from GPIO to sensor 
# and if we have a perfect touch we  should receive the prbs signal back from the sensor (the 7 pins)
# 7 pins are our input,  gpio is our output
# we need to create a function that will drive the prbs signal from GPIO to sense the touch signal

# N is length of the output sequence. Drives 5 GPIO pins with phase-offset PRBS at ~15 kHz.



def drive_prbs_and_sample(N, taps, channel, adc_obj, seq0, seq1, seq2, seq3, seq4):
    """
    Drive PRBS on 5 GPIO pins and sample one ADC channel at each PRBS chip.
    Returns array of N voltage samples (in volts) for that channel.
    seq0..seq4 are pre-generated phase-offset PRBS sequences.
    """
    dt = 1.0 / 15000  # 15 kHz PRBS clock
    # Dummy read so ADC settles on this channel after mux switch (avoids bad first sample)
    adc_obj.ADS1256_SetChannal(channel)
    _ = adc_obj.ADS1256_GetChannalValue(channel)
    samples = []
    for i in range(N):
        GPIO.output(21, int(seq0[i]))
        GPIO.output(20, int(seq1[i]))
        GPIO.output(16, int(seq2[i]))
        GPIO.output(12, int(seq3[i]))
        GPIO.output(7, int(seq4[i]))
        raw = adc_obj.ADS1256_GetChannalValue(channel)
        samples.append(raw * 5.0 / 0x7FFFFF)
        time.sleep(dt)
    return np.array(samples)

# calling

n = 511  # length of the output sequence
t = prbs.taps_prbs511  # taps for the PRBS code
# phase values, space out phase evenly based on length of N to avoid overlap
a = 0
b =int(n/5)
c = int(2*n/5)
d = int(3*n/5)
e = int(4*n/5)

# Pre-generate phase-offset PRBS sequences (once)
seq0 = prbs.prbs_generator(n, t, phase=a)
seq1 = prbs.prbs_generator(n, t, phase=b)
seq2 = prbs.prbs_generator(n, t, phase=c)
seq3 = prbs.prbs_generator(n, t, phase=d)
seq4 = prbs.prbs_generator(n, t, phase=e) 

# Collect 511 ADC samples per channel WHILE driving PRBS (one channel per run)
# Each run: drive PRBS, sample that channel at each chip -> proper time-series for correlation
adc_values = np.zeros((8, n))
for ch in range(8):
    adc_values[ch] = drive_prbs_and_sample(n, t, ch, ADC, seq0, seq1, seq2, seq3, seq4)

# Correlate PRBS (phase 0) with ADC time-series for each channel.
# Use BPSK mapping (-1/+1) for reference so correlation has a clear peak (same as PRBS_code_SG demo).
seq0_bpsk = 2.0 * np.array(seq0, dtype=float) - 1.0
touch_signal = np.array([prbs.corr(seq0_bpsk, adc_values[ch], n) for ch in range(8)])

# within each channel, we need to find the lag that gives the highest correlation
for i in range(8):
    sorted_indices = np.argsort(touch_signal[i])
    top_n_indices = np.sort(sorted_indices[-5:])
    for j in top_n_indices:
        lag, val = j, touch_signal[i][j]
        print(f"Channel {i}, Lag {lag}, Value {val}")   # we need to plot the correlation for each channel

# Save all data to CSV
out_dir = _script_dir
# ADC samples: rows = sample index (0..n-1), cols = channel 0..7
np.savetxt(os.path.join(out_dir, "adc_samples.csv"), adc_values.T, delimiter=",", header="ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7", comments="")
# Correlation: rows = lag (0..n-1), cols = channel 0..7
np.savetxt(os.path.join(out_dir, "correlation.csv"), touch_signal.T, delimiter=",", header="ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7", comments="")
# Peak summary: channel, peak_lag, peak_value
with open(os.path.join(out_dir, "peaks_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["channel", "peak_lag", "peak_value"])
    for i in range(8):
        peak_lag = int(np.argmax(touch_signal[i]))
        peak_val = touch_signal[i][peak_lag]
        w.writerow([i, peak_lag, peak_val])
print(f"Saved: adc_samples.csv, correlation.csv, peaks_summary.csv -> {out_dir}")

# --- Plot 1: Correlation vs lag for each sense channel (like a "good" correlation plot) ---
fig1, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
axes = axes.flatten()
for ch in range(8):
    axes[ch].plot(touch_signal[ch], linewidth=0.8)
    axes[ch].set_title(f"Ch {ch}")
    axes[ch].grid(True, alpha=0.3)
    axes[ch].set_xlabel("Lag")
fig1.suptitle("Correlation (PRBS ref) vs lag per sense channel")
plt.tight_layout()
plt.show()

# --- Plot 2: 2D touch map (sense x drive) for comparison with PDF-style display ---
# Use peak at each drive phase per channel: 7 sense lines, 5 drives -> sample correlation at phase lags
n_drives = 5
shift = n // n_drives
touch_map = np.zeros((8, n_drives))
for ch in range(8):
    for d in range(n_drives):
        lag = d * shift
        touch_map[ch, d] = touch_signal[ch][lag]
fig2, ax2 = plt.subplots()
im = ax2.imshow(touch_map, cmap="viridis", interpolation="nearest", aspect="auto")
ax2.set_xlabel("Drive (phase index)")
ax2.set_ylabel("Sense channel")
ax2.set_title("Touch map (correlation at drive phases)")
plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()
