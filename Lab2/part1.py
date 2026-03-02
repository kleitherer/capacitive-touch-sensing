import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO


@dataclass(frozen=True)
class Config:
    autocorr_plot_path: str
    drive_pins: tuple
    sense_channels: tuple
    polynomial: int
    seed: int


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0,
    os.path.join(
        SCRIPT_DIR,
        "High-Precision-AD-DA-Board-Demo-Code",
        "RaspberryPI",
        "ADS1256",
        "python3",
    ),
)
import ADS1256  # noqa: E402


CFG = Config(
    autocorr_plot_path=os.path.join(SCRIPT_DIR, "autocorr_7ch_latest.png"),
    drive_pins=(21, 7, 12, 16, 20),
    sense_channels=(1, 2, 3, 4, 5, 6, 7),
    polynomial=0xB8,
    seed=0x01,
)


def generate_prbs(polynomial: int, length: int, seed: int) -> np.ndarray:
    num_bits = polynomial.bit_length()
    lfsr = [int(bit) for bit in format(seed, f"0{num_bits}b")]
    seq = []
    for _ in range(length):
        seq.append(lfsr[-1])
        feedback = 0
        for bit_pos in range(num_bits):
            if (polynomial >> bit_pos) & 1:
                feedback ^= lfsr[-(bit_pos + 1)]
        lfsr = lfsr[1:] + [feedback]
    return np.array(seq, dtype=np.int8)


def circular_cross_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = max(len(x), len(y))
    x = np.pad(x, (0, n - len(x)), mode="constant")
    y = np.pad(y, (0, n - len(y)), mode="constant")
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        out[k] = np.sum(x * np.roll(y, k))
    return out


def setup_adc_and_gpio(cfg: Config):
    adc = ADS1256.ADS1256()
    adc.ADS1256_init()
    adc.ADS1256_WriteReg(0x00, 0x02)  # input buffer ON

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    for pin in cfg.drive_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    return adc


def build_drive_sequences(cfg: Config):
    length = (2 ** cfg.polynomial.bit_length()) - 1
    shift = length // len(cfg.drive_pins)
    prbs0 = generate_prbs(cfg.polynomial, length, cfg.seed)
    prbs_matrix = np.vstack([np.roll(prbs0, i * shift) for i in range(len(cfg.drive_pins))])
    return prbs0, prbs_matrix, length


def drive_one_bit(prbs_matrix: np.ndarray, bit_index: int, pins):
    for i, pin in enumerate(pins):
        GPIO.output(pin, int(prbs_matrix[i, bit_index]))


def collect_and_correlate(adc, cfg: Config, prbs0: np.ndarray, prbs_matrix: np.ndarray, length: int):
    raw_sense = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)
    autocorr = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)

    for j, sense_ch in enumerate(cfg.sense_channels):
        adc.ADS1256_SetChannal(sense_ch)
        for s in range(length):
            drive_one_bit(prbs_matrix, s, cfg.drive_pins)
            adc_value = adc.ADS1256_GetChannalValue(sense_ch)
            raw_sense[j, s] = adc_value * 5.0 / 0x7FFFFF

        # "Autocorrelation" style plot used in lab flow: PRBS reference vs sensed waveform.
        autocorr[j, :] = circular_cross_correlation(prbs0.astype(np.float64), raw_sense[j, :])

    return autocorr


def save_autocorr_plot(autocorr: np.ndarray, cfg: Config):
    plt.figure(figsize=(9, 5))
    for i, ch in enumerate(cfg.sense_channels):
        plt.plot(autocorr[i], label=f"Ch {ch}", linewidth=1.2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.title("Autocorrelation-style Correlation for 7 Sense Channels")
    plt.legend(loc="upper right", fontsize=8)
    plt.xlim(0, autocorr.shape[1] - 1)
    plt.savefig(cfg.autocorr_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    adc = setup_adc_and_gpio(CFG)
    prbs0, prbs_matrix, length = build_drive_sequences(CFG)

    frame_count = 0
    start_time = time.time()
    print(f"Saving latest plot to: {CFG.autocorr_plot_path}")

    while True:
        autocorr = collect_and_correlate(adc, CFG, prbs0, prbs_matrix, length)
        save_autocorr_plot(autocorr, CFG)

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frames: {frame_count}, {fps:.2f} Hz")


if __name__ == "__main__":
    main()
