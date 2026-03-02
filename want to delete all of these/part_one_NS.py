"""
Part I 

Implements:
1) ADS1256 bring-up on AD/DA HAT
2) ADC input buffer enabled
3) ADC configured for one gain and one data rate across channels
4) 5 GPIO drive lines output phase-shifted PRBS at 15 kHz
5) ADC sense channels sampled while PRBS is driven
"""

import os
import sys
import time
import numpy as np
import RPi.GPIO as GPIO

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256 

DRIVE_PINS = [7, 12, 16, 20, 21]  # based on lab spec
SENSE_CHANNELS = [1, 2, 3, 4, 5, 6, 7]

PRBS_ORDER = 8                    # change to be btwn 2...16
PRBS_SEED = 0x01                  # initial phase everything is shifted from this
PRBS_FREQ_HZ = 15000.0            # lab specs (15kHz)
N_SKIP = 5                        # number of samples to skip (settling time, needed 2 add bc of noise) 

# gain/rate for all channels
ADC_GAIN_KEY = "ADS1256_GAIN_32"
ADC_DRATE_KEY = "ADS1256_30000SPS"

# taps mask for PRBS8
PRBS_TAPS_MASK = 0xB8

# setting up the GPIO pins/ADC
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in DRIVE_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

adc = ADS1256.ADS1256()
adc.ADS1256_init()
adc.ADS1256_WriteReg(0x00, 0x02)
adc.ADS1256_ConfigADC(
    ADS1256.ADS1256_GAIN_E[ADC_GAIN_KEY],
    ADS1256.ADS1256_DRATE_E[ADC_DRATE_KEY],
)

def lfsr_prbs(order: int, taps_mask: int, seed: int) -> np.ndarray:
    length = (1 << order) - 1
    state = seed & ((1 << order) - 1)

    seq = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        seq[i] = state & 0x1

        tapped = state & taps_mask
        parity = 0
        while tapped:
            parity ^= tapped & 0x1
            tapped >>= 1

        state = (state >> 1) | (parity << (order - 1))
        state &= (1 << order) - 1

    return seq


def build_phase_shifted_prbs() -> np.ndarray:
    """returns array shape (5, N): same PRBS but with 5 distinct phase offsets."""
    base = lfsr_prbs(PRBS_ORDER, PRBS_TAPS_MASK, PRBS_SEED)
    n = len(base)
    shift = n // 5
    return np.vstack([
        np.roll(base, 0 * shift),
        np.roll(base, 1 * shift),
        np.roll(base, 2 * shift),
        np.roll(base, 3 * shift),
        np.roll(base, 4 * shift),
    ])


def drive_and_sample(adc, seq_matrix: np.ndarray) -> np.ndarray:
    """capture one PRBS-length for each sense ch."""
    n = seq_matrix.shape[1]
    dt = 1.0 / PRBS_FREQ_HZ
    samples = np.zeros((len(SENSE_CHANNELS), n), dtype=np.float64)

    for ch_i, ch in enumerate(SENSE_CHANNELS):
        temp = []
        total = n + N_SKIP
        for i in range(total):
            t0 = time.perf_counter()
            j = i - N_SKIP if i >= N_SKIP else i

            for pin_i, pin in enumerate(DRIVE_PINS):
                GPIO.output(pin, int(seq_matrix[pin_i, j]))

            raw = adc.ADS1256_GetChannalValue(ch)
            temp.append(raw * 5.0 / 0x7FFFFF)

            while time.perf_counter() - t0 < dt:
                pass

        samples[ch_i, :] = np.array(temp[N_SKIP:], dtype=np.float64)

    return samples


seq_matrix = build_phase_shifted_prbs()
n = seq_matrix.shape[1]

print("part1")
print(f"PRBS length: {n} (order={PRBS_ORDER})")

try:
    t_start = time.perf_counter()
    data = drive_and_sample(adc, seq_matrix)
    elapsed = time.perf_counter() - t_start
    print(f"Captured shape {data.shape} in {elapsed:.3f} s")

    out_csv = os.path.join(_script_dir, "part_one_clean_samples.csv")
    header = ",".join([f"ch{c}" for c in SENSE_CHANNELS])
    np.savetxt(out_csv, data.T, delimiter=",", header=header, comments="")
    print(f"Saved: {out_csv}")
finally:
    for pin in DRIVE_PINS:
        GPIO.output(pin, 0)
    GPIO.cleanup(DRIVE_PINS)
