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

# ADC operating point for all channels (single global setting).
# Use the highest gain that does not clip on touch events.
ADC_GAIN_KEY = 'ADS1256_GAIN_32'
ADC_DRATE_KEY = 'ADS1256_30000SPS'


# we're setting the drive lines up ourselves ... so we have to use readADCdata
ADC = ADS1256.ADS1256()
ADC.ADS1256_init()
# ADS1256 STATUS register: enable analog input buffer (BUFEN bit = 1 -> 0x02)
ADC.ADS1256_WriteReg(0x00, 0x02)
# Re-apply ADC gain/data-rate explicitly so all channels use the same gain.
ADC.ADS1256_ConfigADC(
    ADS1256.ADS1256_GAIN_E[ADC_GAIN_KEY],
    ADS1256.ADS1256_DRATE_E[ADC_DRATE_KEY]
)
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



# PRBS clock per lab spec: 15 kHz with ADC at 30 kSPS.
PRBS_FREQ = 15 * 10**3  # Hz

# Discard first N samples after channel switch (mux settling). GetChannalValue does not
# add settling delay; ADS1256 typically needs ~4-5 conversion cycles after mux change.
N_SKIP_SETTLING = 5

# Average this many full PRBS captures per channel to suppress random noise.
N_AVG = 8

def normalized_circular_corr(reference, signal):
    """
    Signed, normalized circular cross-correlation.
    Returns values in approximately [-1, 1], with low noise floor around 0.
    """
    ref = np.asarray(reference, dtype=float)
    sig = np.asarray(signal, dtype=float)
    ref = ref - np.mean(ref)
    sig = sig - np.mean(sig)
    denom = np.linalg.norm(ref) * np.linalg.norm(sig)
    if denom < 1e-12:
        return np.zeros(len(ref))

    corr = np.zeros(len(ref))
    for k in range(len(ref)):
        corr[k] = np.sum(ref * np.roll(sig, k)) / denom
    return corr

def drive_prbs_and_sample(N, taps, channel, adc_obj, seq0, seq1, seq2, seq3, seq4):
    """
    Drive PRBS on 5 GPIO pins and sample one ADC channel at each PRBS chip.
    Returns array of N voltage samples (in volts) for that channel.
    seq0..seq4 are pre-generated phase-offset PRBS sequences.
    At 30 kHz: no sleep - ADC conversion (~33 µs) is the rate limiter; minimizes jitter.
    At 15 kHz: spin-wait (not sleep) for remaining time - sleep() has poor sub-ms resolution.
    Discards first N_SKIP_SETTLING samples after channel switch (mux settling).
    """
    target_period = 1.0 / PRBS_FREQ
    samples = []
    total = N + N_SKIP_SETTLING
    for i in range(total):
        loop_start = time.perf_counter()
        j = i - N_SKIP_SETTLING if i >= N_SKIP_SETTLING else i  # kept samples align with seq[0..N-1]
        GPIO.output(21, int(seq0[j]))
        GPIO.output(20, int(seq1[j]))
        GPIO.output(16, int(seq2[j]))
        GPIO.output(12, int(seq3[j]))
        GPIO.output(7, int(seq4[j]))
        raw = adc_obj.ADS1256_GetChannalValue(channel)
        samples.append(raw * 5.0 / 0x7FFFFF)
        # Spin-wait for remaining time (sleep() is unreliable for sub-ms intervals)
        while time.perf_counter() - loop_start < target_period:
            pass
    return np.array(samples[N_SKIP_SETTLING:])  # discard settling samples

# calling

n = 255  # length of the output sequence (PRBS-8)
t = prbs.taps_prbs255  # taps for the PRBS code
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

# Collect ADC samples per channel WHILE driving PRBS (one channel per run)
# Each run: drive PRBS, sample that channel at each chip -> proper time-series for correlation
# Use ADC channels 1-7 (sense lines), not channel 0
SENSE_CHANNELS = list(range(1, 8))  # [1, 2, 3, 4, 5, 6, 7]
adc_values = np.zeros((7, n))
seq0_bpsk = 2 * seq0 - 1  # map 0->-1, 1->+1
touch_signal = np.zeros((7, n))
for i, ch in enumerate(SENSE_CHANNELS):
    adc_acc = np.zeros(n)
    corr_acc = np.zeros(n)
    for _ in range(N_AVG):
        run = drive_prbs_and_sample(n, t, ch, ADC, seq0, seq1, seq2, seq3, seq4)
        adc_acc += run
        corr_acc += normalized_circular_corr(seq0_bpsk, run)
    adc_values[i] = adc_acc / N_AVG
    touch_signal[i] = corr_acc / N_AVG

# within each channel, we need to find the lag that gives the highest correlation
for i in range(7):
    sorted_indices = np.argsort(touch_signal[i])
    top_n_indices = np.sort(sorted_indices[-5:])
    for j in top_n_indices:
        lag, val = j, touch_signal[i][j]
        print(f"Channel {SENSE_CHANNELS[i]}, Lag {lag}, Value {val}")   # we need to plot the correlation for each channel

# # Save all data to CSV
out_dir = _script_dir
# # ADC samples: rows = sample index (0..n-1), cols = channel 0..6
# np.savetxt(os.path.join(out_dir, "adc_samples.csv"), adc_values.T, delimiter=",", header="ch1,ch2,ch3,ch4,ch5,ch6,ch7", comments="")
# # Correlation: rows = lag (0..n-1), cols = channel 0..6
# np.savetxt(os.path.join(out_dir, "correlation.csv"), touch_signal.T, delimiter=",", header="ch1,ch2,ch3,ch4,ch5,ch6,ch7", comments="")
# # Peak summary: channel, peak_lag, peak_value
# with open(os.path.join(out_dir, "peaks_summary.csv"), "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["channel", "peak_lag", "peak_value"])
#     for i in range(7):
#         peak_lag = int(np.argmax(touch_signal[i]))
#         peak_val = touch_signal[i][peak_lag]
#         w.writerow([SENSE_CHANNELS[i], peak_lag, peak_val])
# print(f"Saved: adc_samples.csv, correlation.csv, peaks_summary.csv -> {out_dir}")

for i in range(7):
    plt.plot(touch_signal[i], label=f"Channel {SENSE_CHANNELS[i]}")
plt.grid()
plt.xlabel("Lag")
plt.ylabel("Normalized correlation")
plt.title("Normalized circular correlation: ADC waveform vs PRBS255")
plt.xlim(0, n - 1)
plt.ylim(-1.0, 1.0)
plt.legend()
plt.savefig(os.path.join(out_dir, "autocorrelation_plot.png"), dpi=150)
print(f"Saved: autocorrelation_plot.png -> {out_dir}")
plt.show()