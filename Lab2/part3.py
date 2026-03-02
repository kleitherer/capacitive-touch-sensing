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
    save_path=os.path.join(SCRIPT_DIR, "part3_heatmap_latest.png"),
    centroid_path=os.path.join(SCRIPT_DIR, "part3_centroid_ellipse.png"),
    drive_pins=(21, 7, 12, 16, 20),
    sense_channels=(1, 2, 3, 4, 5, 6, 7),
    polynomial=0xB8,
    seed=0x01,
    threshold=200.0,
)


class KalmanXYVelocity:
    """
    Constant-velocity Kalman filter with state:
      x = [pos_x, pos_y, vel_x, vel_y]^T
    Measurement:
      z = [pos_x, pos_y]^T
    """
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2, dtype=float) * 0.20
        self.initialized = False

    def _F(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def _Q(self, dt: float, sigma_a: float = 2.0) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = sigma_a * sigma_a
        return q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )

    def initialize(self, x_meas: float, y_meas: float):
        self.x = np.array([[x_meas], [y_meas], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float) * 1.0
        self.initialized = True

    def predict(self, dt: float):
        if not self.initialized:
            return
        F = self._F(dt)
        Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, x_meas: float, y_meas: float):
        if not self.initialized:
            self.initialize(x_meas, y_meas)
            return
        z = np.array([[x_meas], [y_meas]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def state(self):
        if not self.initialized:
            return None
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])


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
    del threshold
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


def draw_ellipse(ax, result, color: str):
    if result is None:
        return
    cx, cy, major, minor, angle = result
    ax.add_patch(
        Ellipse(
            (cx, cy),
            major,
            minor,
            angle=angle,
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
    )


def setup_adc_and_gpio(cfg: Config):
    adc = ADS1256.ADS1256()
    adc.ADS1256_init()
    adc.ADS1256_WriteReg(0x00, 0x02)

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


def compute_touch_map(adc, cfg: Config, prbs0: np.ndarray, prbs_matrix: np.ndarray, length: int, shift: int):
    raw_sense = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)
    xcor_raw = np.zeros((len(cfg.sense_channels), length), dtype=np.float64)
    xcor = np.zeros((len(cfg.sense_channels), len(cfg.drive_pins)), dtype=np.float64)

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


def apply_axis_format(ax, cfg: Config):
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Drive line")
    ax.set_ylabel("Channel")
    ax.set_xticks(np.arange(len(cfg.drive_pins)))
    ax.set_xticklabels([str(i) for i in range(len(cfg.drive_pins), 0, -1)])
    ax.set_yticks(np.arange(len(cfg.sense_channels)))
    ax.set_yticklabels([str(i) for i in range(len(cfg.sense_channels), 0, -1)])


def save_outputs(xcor_plot: np.ndarray, cfg: Config, raw_result, kf_state):
    # Heatmap with raw ellipse + filtered centroid/velocity
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax_h = max(cfg.threshold, np.max(xcor_plot), 1)
    im = ax.imshow(xcor_plot, cmap="RdPu", interpolation="nearest", vmin=0, vmax=vmax_h)
    fig.colorbar(im, ax=ax)
    apply_axis_format(ax, cfg)

    draw_ellipse(ax, raw_result, "white")
    if raw_result is not None:
        ax.plot(raw_result[0], raw_result[1], "w+", markersize=10, markeredgewidth=2, label="raw")

    if kf_state is not None:
        fx, fy, vx, vy = kf_state
        ax.plot(fx, fy, "co", markersize=7, label="kalman")
        speed = float(np.hypot(vx, vy))
        ax.text(
            0.02,
            0.98,
            f"Filtered x,y: ({fx:.2f}, {fy:.2f})\nVx,Vy: ({vx:.2f}, {vy:.2f})\nSpeed: {speed:.2f} cells/s",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"),
        )
    ax.set_title("Part 3: Heatmap + Raw Ellipse + Kalman Centroid")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Centroid map view
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    vmax_plot = max(cfg.threshold, np.max(xcor_plot), 1)
    ax2.imshow(xcor_plot, cmap="magma", interpolation="nearest", vmin=0, vmax=vmax_plot)
    apply_axis_format(ax2, cfg)
    draw_ellipse(ax2, raw_result, "white")
    if raw_result is not None:
        ax2.plot(raw_result[0], raw_result[1], "w+", markersize=10, markeredgewidth=2)
    if kf_state is not None:
        fx, fy, vx, vy = kf_state
        ax2.plot(fx, fy, "co", markersize=7)
        ax2.set_title(f"Kalman centroid ({fx:.2f}, {fy:.2f}) | V=({vx:.2f},{vy:.2f})")
    else:
        ax2.set_title("No touch detected")
    fig2.savefig(cfg.centroid_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def main():
    adc = setup_adc_and_gpio(CFG)
    prbs0, prbs_matrix, length, shift = build_drive_sequences(CFG)
    kf = KalmanXYVelocity()

    frame_count = 0
    start_time = time.time()
    last_t = time.time()

    while True:
        xcor = compute_touch_map(adc, CFG, prbs0, prbs_matrix, length, shift)
        xcor_plot = xcor.copy()
        xcor_plot[xcor_plot < CFG.threshold] = 0

        raw_result = centroid_and_ellipse(xcor_plot, CFG.threshold)

        now_t = time.time()
        dt = max(1e-3, now_t - last_t)
        last_t = now_t

        kf.predict(dt)
        if raw_result is not None:
            raw_cx, raw_cy = raw_result[0], raw_result[1]
            kf.update(raw_cx, raw_cy)
        kf_state = kf.state()

        save_outputs(xcor_plot, CFG, raw_result, kf_state)

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            if kf_state is not None:
                _, _, vx, vy = kf_state
                print(f"Frames: {frame_count}, {fps:.2f} Hz | velocity=({vx:.2f}, {vy:.2f}) cells/s")
            else:
                print(f"Frames: {frame_count}, {fps:.2f} Hz | velocity=(n/a)")


if __name__ == "__main__":
    main()
