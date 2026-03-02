import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import RPi.GPIO as GPIO


@dataclass(frozen=True)
class Config:
    save_path: str
    centroid_path: str
    drive_pins: tuple
    sense_channels: tuple
    polynomial: int
    seed: int
    threshold: float


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256 


CFG = Config(
    save_path=os.path.join(SCRIPT_DIR, "heatmap_latest.png"),
    centroid_path=os.path.join(SCRIPT_DIR, "centroid_ellipse.png"),
    drive_pins=(21, 7, 12, 16, 20),   
    sense_channels=(1, 2, 3, 4, 5, 6, 7),
    polynomial=0xB8,
    seed=0x01,
    threshold=200.0,
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


def centroid_and_ellipse(data: np.ndarray, threshold: float):
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
    return cx, cy, major, minor, angle


def draw_centroid_overlay(ax, result):
    if result is None:
        return
    cx, cy, major, minor, angle = result
    ax.plot(cx, cy, "w+", markersize=12, markeredgewidth=2)
    ellipse = Ellipse(
        (cx, cy),
        major,
        minor,
        angle=angle,
        fill=False,
        edgecolor="white",
        linewidth=2,
    )
    ax.add_patch(ellipse)


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
    return prbs0, prbs_matrix, length, shift


def drive_one_bit(prbs_matrix: np.ndarray, bit_index: int, pins):
    for i, pin in enumerate(pins):
        GPIO.output(pin, int(prbs_matrix[i, bit_index]))


def compute_touch_maps(adc, cfg: Config, prbs0: np.ndarray, prbs_matrix: np.ndarray, length: int, shift: int):
    raw_sense = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)
    xcor_raw = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)
    xcor = np.zeros((len(cfg.sense_channels), len(cfg.drive_pins)), dtype=np.float64)

    # Sequentially scan each sense line; PRBS starts from bit 0 for each channel.
    for j, sense_ch in enumerate(cfg.sense_channels):
        adc.ADS1256_SetChannal(sense_ch)
        for s in range(length):
            drive_one_bit(prbs_matrix, s, cfg.drive_pins)
            adc_value = adc.ADS1256_GetChannalValue(sense_ch)
            raw_sense[j, s] = adc_value * 5.0 / 0x7FFFFF

        xcor_raw[j, :] = circular_cross_correlation(prbs0.astype(np.float64), raw_sense[j, :])
        xcor[j, :] = [
            xcor_raw[j, 0],
            xcor_raw[j, 1 * shift],
            xcor_raw[j, 2 * shift],
            xcor_raw[j, 3 * shift],
            xcor_raw[j, 4 * shift],
        ]

    return xcor


def save_heatmap_image(xcor_plot: np.ndarray, cfg: Config):
    result = centroid_and_ellipse(xcor_plot, cfg.threshold)
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax_h = max(cfg.threshold, np.max(xcor_plot), 1)
    im = ax.imshow(xcor_plot, cmap="RdPu", interpolation="nearest", vmin=0, vmax=vmax_h)
    fig.colorbar(im, ax=ax)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Drive line")
    ax.set_ylabel("Channel")
    ax.set_xticks(np.arange(len(cfg.drive_pins)))
    ax.set_xticklabels([str(i) for i in range(len(cfg.drive_pins), 0, -1)])
    ax.set_yticks(np.arange(len(cfg.sense_channels)))
    ax.set_yticklabels([str(i) for i in range(len(cfg.sense_channels), 0, -1)])
    draw_centroid_overlay(ax, result)
    if result is not None:
        cx, cy, major, minor, _ = result
        ax.set_title(f"Centroid ({cx:.2f}, {cy:.2f})")
        ax.text(
            0.02,
            0.98,
            f"Major: {major:.2f}\nMinor: {minor:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"),
        )
    else:
        ax.set_title("")
    fig.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_centroid_image(xcor_plot: np.ndarray, cfg: Config):
    result = centroid_and_ellipse(xcor_plot, cfg.threshold)
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax_plot = max(cfg.threshold, np.max(xcor_plot), 1)
    ax.imshow(xcor_plot, cmap="magma", interpolation="nearest", vmin=0, vmax=vmax_plot)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Drive line")
    ax.set_ylabel("Channel")
    ax.set_xticks(np.arange(len(cfg.drive_pins)))
    ax.set_xticklabels([str(i) for i in range(len(cfg.drive_pins), 0, -1)])
    ax.set_yticks(np.arange(len(cfg.sense_channels)))
    ax.set_yticklabels([str(i) for i in range(len(cfg.sense_channels), 0, -1)])

    if result is not None:
        cx, cy, major, minor, angle = result
        draw_centroid_overlay(ax, result)
        ax.set_title(f"Centroid ({cx:.2f}, {cy:.2f}) | Major={major:.2f} Minor={minor:.2f}")
    else:
        ax.set_title("No touch detected")

    plt.savefig(cfg.centroid_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    adc = setup_adc_and_gpio(CFG)
    prbs0, prbs_matrix, length, shift = build_drive_sequences(CFG)

    frame_count = 0
    start_time = time.time()

    while True:
        xcor = compute_touch_maps(adc, CFG, prbs0, prbs_matrix, length, shift)
        xcor_plot = xcor.copy()
        xcor_plot[xcor_plot < CFG.threshold] = 0

        save_heatmap_image(xcor_plot, CFG)
        save_centroid_image(xcor_plot, CFG)

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frames: {frame_count}, {fps:.2f} Hz")


if __name__ == "__main__":
    main()
