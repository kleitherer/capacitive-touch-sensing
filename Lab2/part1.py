import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO

# add ADS1256 driver folder path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256

# PRBS tap polynomials (hex). Length = 2^nbits - 1 for nbits from tap width.
taps_prbs7 = 0x6
taps_prbs15 = 0xC
taps_prbs31 = 0x14
taps_prbs63 = 0x30
taps_prbs127 = 0x60
taps_prbs255 = 0xB8
taps_prbs511 = 0x110
taps_prbs1023 = 0x240

# using class bc easier to copy/paste to part2 and part3
# got idea for this from Cursor
@dataclass(frozen=True)
class Config:
    autocorr_plot_path: str
    drive_pins: tuple
    sense_channels: tuple
    taps: int
    phase: int

CFG = Config(
    autocorr_plot_path=os.path.join(SCRIPT_DIR, "autocorr_7ch_latest.png"),
    # bcm pins from the lab wiring
    drive_pins=(21, 7, 12, 16, 20),
    # these are the 7 receive/sense lines we scan one by one
    sense_channels=(1, 2, 3, 4, 5, 6, 7),
    taps=taps_prbs255,  # PRBS-255 (8-bit); use taps_prbs511 etc. for other lengths
    phase=0x01,
)


def generate_prbs(taps, length, phase=0, initial_value=None):
    """
    Generate one PRBS sequence from LFSR (same logic from HW3)
    taps: polynomial in hex (e.g. 0xB8 for PRBS-8, 0x110 for PRBS-511).
    length: sequence length N (e.g. 255 for 8-bit, 511 for 9-bit).
    phase: circular shift applied to output (for orthogonal drive phases).
    initial_value: optional initial LFSR state (array of length nbits); default ones.
    """
    nbits = int(np.log2(length + 1))  # N = 2^nbits - 1
    if initial_value is not None:
        if np.isscalar(initial_value):
            seq_values = np.array([int(b) for b in format(int(initial_value), f"0{nbits}b")], dtype=int)
        else:
            seq_values = np.asarray(initial_value, dtype=int)
    else:
        seq_values = np.ones(nbits, dtype=int)

    taps_array = np.array([int(b) for b in format(taps, f"0{nbits}b")])
    ones_indices = np.where(taps_array == 1)[0]
    xor_indices = [nbits - 1 - ones_indices[i] for i in range(len(ones_indices))]

    output = np.zeros(length, dtype=np.float64)
    for i in range(length):
        output[i] = seq_values[-1]
        vals_to_xor = seq_values[xor_indices]
        xor_result = np.bitwise_xor.reduce(vals_to_xor)
        seq_values = np.roll(seq_values, 1)
        seq_values[0] = xor_result

    if phase != 0:
        output = np.roll(output, int(phase))
    return np.array(output, dtype=np.int8)


def circular_cross_correlation(x, y):
    n = max(len(x), len(y))
    x = np.pad(x, (0, n - len(x)), mode="constant")
    y = np.pad(y, (0, n - len(y)), mode="constant")
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        out[k] = np.sum(x * np.roll(y, k))
    return out


def setup_adc_and_gpio(cfg):
    adc = ADS1256.ADS1256()
    adc.ADS1256_init()
    adc.ADS1256_WriteReg(0x00, 0x02)  # this is what tells us our input buffer is ON

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    for pin in cfg.drive_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    return adc


def build_drive_sequences(cfg: Config):
    """
    One reference PRBS (prbs0) plus a matrix of phase-shifted copies, 
    one row per drive line, so correlations are separable by drive.
    """
    # N = 2^m - 1 from tap bit length
    length = (2 ** cfg.taps.bit_length()) - 1
    # each drive gets same prbs but shifted phase
    shift = length // len(cfg.drive_pins)
    prbs0 = generate_prbs(cfg.taps, length, cfg.phase)
    prbs_rows = []
    for i in range(len(cfg.drive_pins)):
        row = np.roll(prbs0, i * shift)
        prbs_rows.append(row)
    prbs_matrix = np.vstack(prbs_rows)
    return prbs0, prbs_matrix, length


def drive_one_bit(prbs_matrix, bit_index, pins):
    for i, pin in enumerate(pins):
        GPIO.output(pin, int(prbs_matrix[i, bit_index]))


def collect_and_correlate(adc, cfg, prbs0, prbs_matrix, length):
    """
    We need to record readings from ADC for each sense line and do it sequentially
    Then we run one full PRBS and correlate it with each sense line reading to see lag vs correlation.

    To make things easier, restart the PRBS sequence after switching the ADC to a new sense line (that way, 
    the PRBS sequence timing is exactly the same for every sensing frame). 
    """
    raw_sense = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)
    autocorr = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)

    # can only read one adc channel at a time
    for j, sense_ch in enumerate(cfg.sense_channels):
        adc.ADS1256_SetChannal(sense_ch)
        # run through full prbs for this one sense channel
        for s in range(length):
            drive_one_bit(prbs_matrix, s, cfg.drive_pins)
            adc_value = adc.ADS1256_GetChannalValue(sense_ch)
            raw_sense[j, s] = adc_value * 5.0 / 0x7FFFFF

        # correlate with prbs reference
        autocorr[j, :] = circular_cross_correlation(prbs0.astype(np.float64), raw_sense[j, :])

    return autocorr


def main():
    # initialize everything
    adc = setup_adc_and_gpio(CFG)
    prbs0, prbs_matrix, length = build_drive_sequences(CFG)

    frame_count = 0
    start_time = time.time()
    print(f"save path {CFG.autocorr_plot_path}")

    # we need to keep updating plot image while script runs
    while True:
        autocorr = collect_and_correlate(adc, CFG, prbs0, prbs_matrix, length)
        plt.figure(figsize=(9, 5))
        for i, ch in enumerate(CFG.sense_channels):
            plt.plot(autocorr[i], label=f"Ch {ch}", linewidth=1.2)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.title("Autocorrelation-style Correlation for 7 Sense Channels")
        plt.legend(loc="upper right", fontsize=8)
        plt.xlim(0, autocorr.shape[1] - 1)
        plt.savefig(CFG.autocorr_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frames: {frame_count}, {fps:.2f} Hz")


if __name__ == "__main__":
    main()
