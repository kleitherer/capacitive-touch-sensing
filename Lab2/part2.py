import os
import sys
import time
import numpy as np

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap_latest.png")

import matplotlib
matplotlib.use('Agg')  # No display needed; run headless on Pi
import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256
import RPi.GPIO as GPIO

def generate_prbs(polynomial, length, seed):
    """
    Generate a Pseudo-Random Bit Sequence (PRBS) using an LFSR defined by `polynomial`.
    polynomial: bitmask of taps (example 0xb8)
    length: PRBS length (example 2**8-1)
    seed: non-zero initial state
    """
    num_bits = polynomial.bit_length()
    lfsr = [int(bit) for bit in format(seed, f'0{num_bits}b')]
    prbs_sequence = []
    for _ in range(length):
        prbs_sequence.append(lfsr[-1])
        feedback = 0
        for bit_position in range(num_bits):
            if (polynomial >> bit_position) & 1:
                feedback ^= lfsr[-(bit_position + 1)]
        lfsr = lfsr[1:] + [feedback]
    return np.array(prbs_sequence, dtype=np.int8)

def circular_cross_correlation(x, y):
    """
    Circular cross-correlation for arbitrary length vectors.
    Returns length N where N = max(len(x), len(y)).
    """
    N = max(len(x), len(y))
    x = np.pad(x, (0, N - len(x)), mode='constant')
    y = np.pad(y, (0, N - len(y)), mode='constant')
    result = np.zeros(N, dtype=np.float64)
    for k in range(N):
        result[k] = np.sum(x * np.roll(y, k))
    return result

# Drive and sense line pins as wired
DRIVE = [21, 7, 12, 16, 20]
SENSE = [1, 2, 3, 4, 5, 6, 7]

def drive_prbs_bit(prbs_matrix, s):
    for i, pin in enumerate(DRIVE):
        GPIO.output(pin, int(prbs_matrix[i, s]))


if __name__ == '__main__':
    # ---- ADC setup ----
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()

    # Enable input buffer (per your existing code)
    ADDR = 0x00
    ON = 0x02
    ADC.ADS1256_WriteReg(ADDR, ON)

    # ---- GPIO setup ----
    # GPIO.setmode(GPIO.BOARD)  # enable if these are BOARD numbers; otherwise use BCM
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)      # <-- adjust if your pin numbers are BOARD
    for pin in DRIVE:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    # ---- PRBS setup ----
    polynomial = 0xb8
    length = 2**8 - 1
    shift = length // 5
    seed = 0x01

    prbs0 = generate_prbs(polynomial, length, seed)
    prbs1 = np.roll(prbs0, 1 * shift)
    prbs2 = np.roll(prbs0, 2 * shift)
    prbs3 = np.roll(prbs0, 3 * shift)
    prbs4 = np.roll(prbs0, 4 * shift)
    prbs = np.vstack([prbs0, prbs1, prbs2, prbs3, prbs4])  # shape (5, length)

    # ---- Storage ----
    raw_sense = np.zeros((len(SENSE), length), dtype=np.float64)

    # correlation outputs:
    # xcor_raw[j, :] is correlation vs lag for sense j
    # xcor[j, i] is correlation at the phase taps for each drive i
    xcor_raw = np.zeros((len(SENSE), length), dtype=np.float64)
    xcor = np.zeros((len(SENSE), len(DRIVE)), dtype=np.float64)

    threshold = 200

    frame_count = 0
    start_time = time.time()

    while True:
        # === ONE FULL FRAME ===
        # Requirement: scan each sense line sequentially,
        # and restart PRBS (s=0) after switching the ADC mux.

        for j, sense_ch in enumerate(SENSE):
            # Switch ADC input mux to this sense line
            ADC.ADS1256_SetChannal(sense_ch)

            # Restart PRBS for this sense line
            # (this is what your lab text explicitly suggests)
            for s in range(length):
                # Drive PRBS bit on all 5 drive lines
                drive_prbs_bit(prbs, s)

                # Read ADC
                ADC_Value = ADC.ADS1256_GetChannalValue(sense_ch)
                raw_sense[j, s] = ADC_Value * 5.0 / 0x7fffff

                # Optional pacing (usually you won’t actually hit 15kHz in Python)
                # loop_duration = time.time() - loop_start
                # if loop_duration < target_period:
                #     time.sleep(target_period - loop_duration)

            # After collecting full PRBS-length record for THIS sense line,
            # correlate it against prbs0 and pick off the 5 orthogonal phases
            xcor_raw[j, :] = circular_cross_correlation(prbs0.astype(np.float64), raw_sense[j, :])

            xcor[j, :] = [
                xcor_raw[j, 0],
                xcor_raw[j, 1 * shift],
                xcor_raw[j, 2 * shift],
                xcor_raw[j, 3 * shift],
                xcor_raw[j, 4 * shift],
            ]

        xcor_plot = xcor.copy()
        xcor_plot[xcor_plot < threshold] = 0

        plt.figure(figsize=(6, 5))
        plt.imshow(xcor_plot, cmap='viridis', interpolation='nearest', vmin=threshold)
        plt.colorbar()
        plt.title("Heatmap (Sense x Drive)")
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
        plt.close()

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frames: {frame_count}, {fps:.2f} Hz")
