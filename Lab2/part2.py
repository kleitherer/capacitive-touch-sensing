import os
import sys
import time
import numpy as np

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap_latest.png")
CENTROID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "centroid_ellipse.png")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


def centroid_and_ellipse(data, threshold):
    """
    Compute weighted centroid and fit ellipse to touch region.
    data: (n_sense, n_drive) heatmap (values below threshold should be 0)
    Returns: (cx, cy, major_axis, minor_axis, angle_deg) or None if no touch
    """
    w = np.maximum(data, 0)
    total = np.sum(w)
    if total < 1e-9:
        return None

    n_sense, n_drive = data.shape
    yy, xx = np.mgrid[0:n_sense, 0:n_drive]
    cx = np.sum(xx * w) / total
    cy = np.sum(yy * w) / total

    dx = xx - cx
    dy = yy - cy
    cov_xx = np.sum(w * dx * dx) / total
    cov_yy = np.sum(w * dy * dy) / total
    cov_xy = np.sum(w * dx * dy) / total
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-9)
    major = 2 * np.sqrt(eigvals[1])
    minor = 2 * np.sqrt(eigvals[0])
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    return (cx, cy, major, minor, angle)


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

        # Heatmap plot
        plt.figure(figsize=(6, 5))
        vmax_h = max(threshold, np.max(xcor_plot), 1)
        plt.imshow(xcor_plot, cmap='viridis', interpolation='nearest', vmin=0, vmax=vmax_h)
        plt.colorbar()
        plt.title("Heatmap (Sense x Drive)")
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
        plt.close()

        # Centroid + ellipse plot
        result = centroid_and_ellipse(xcor_plot, threshold)
        fig, ax = plt.subplots(figsize=(6, 5))
        vmin_plot = 0
        vmax_plot = max(threshold, np.max(xcor_plot), 1)
        ax.imshow(xcor_plot, cmap='viridis', interpolation='nearest', vmin=vmin_plot, vmax=vmax_plot)
        ax.invert_yaxis()
        ax.invert_xaxis()
        if result is not None:
            cx, cy, major, minor, angle = result
            ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
            ellipse = Ellipse((cx, cy), major, minor, angle=angle, fill=False,
                              edgecolor='red', linewidth=2)
            ax.add_patch(ellipse)
            ax.set_title(f"Centroid ({cx:.2f}, {cy:.2f}) | Major={major:.2f} Minor={minor:.2f}")
        else:
            ax.set_title("No touch detected")
        plt.savefig(CENTROID_PATH, dpi=150, bbox_inches='tight')
        plt.close()

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frames: {frame_count}, {fps:.2f} Hz")
