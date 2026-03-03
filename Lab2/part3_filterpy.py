"""
Part 3 variant: uses the KalmanFilter from kalman_filter.py (filterpy-style)
instead of the custom KalmanXYVelocity class.

Same behavior: constant-velocity 2D tracker, centroid in, smoothed position + velocity out.
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import part1 as p1
import part2 as p2
from kalman_filter import KalmanFilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass(frozen=True)
class Config:
    save_path: str
    drive_pins: tuple
    sense_channels: tuple
    taps: int
    phase: int
    threshold: float

CFG = Config(
    save_path=os.path.join(SCRIPT_DIR, "part3_heatmap_latest.png"),
    drive_pins=(21, 7, 12, 16, 20),
    sense_channels=(1, 2, 3, 4, 5, 6, 7),
    taps=p1.taps_prbs255,  # match part1/part2 tap set
    phase=0x01,
    threshold=200.0,
)

# Kalman tuning (same as original part3)
MEASUREMENT_NOISE_R = 0.20
PROCESS_NOISE_SIGMA_A = 2.0
INITIAL_P = 10.0
INITIAL_P_AFTER_FIRST_MEAS = 1.0


def state_transition_F(dt):
    """Constant-velocity F: state [pos_x, pos_y, vel_x, vel_y]."""
    return np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def process_noise_Q(dt, sigma_a=PROCESS_NOISE_SIGMA_A):
    """Q for continuous white noise acceleration model (dim_x=4)."""
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


def make_touch_kalman():
    """Build and return a KalmanFilter for 2D position + velocity (dim_x=4, dim_z=2)."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.zeros((4, 1), dtype=float)
    kf.P = np.eye(4, dtype=float) * INITIAL_P
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    kf.R = np.eye(2, dtype=float) * MEASUREMENT_NOISE_R
    # F and Q are set per predict(dt) so we don't set kf.F / kf.Q here
    return kf


def draw_ellipse(ax, result, color):
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


def apply_axis_format(ax, cfg):
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Drive line")
    ax.set_ylabel("Channel")
    ax.set_xticks(np.arange(len(cfg.drive_pins)))
    ax.set_xticklabels([str(i) for i in range(len(cfg.drive_pins), 0, -1)])
    ax.set_yticks(np.arange(len(cfg.sense_channels)))
    ax.set_yticklabels([str(i) for i in range(len(cfg.sense_channels), 0, -1)])


def save_outputs(xcor_plot, cfg, raw_result, kf_state):
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
    ax.set_title("Part 3 (filterpy): Heatmap + Raw Ellipse + Kalman Centroid")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8)
    fig.savefig(cfg.save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    adc = p1.setup_adc_and_gpio(CFG)
    prbs0, prbs_matrix, length, shift = p2.build_drive_sequences(CFG)
    kf = make_touch_kalman()
    initialized = False

    frame_count = 0
    start_time = time.time()
    last_t = time.time()

    while True:
        xcor = p2.compute_touch_maps(adc, CFG, prbs0, prbs_matrix, length, shift)
        xcor_plot = xcor.copy()
        xcor_plot[xcor_plot < CFG.threshold] = 0

        raw_result = p2.centroid_and_ellipse(xcor_plot, CFG.threshold)

        now_t = time.time()
        dt = max(1e-3, now_t - last_t)
        last_t = now_t

        if not initialized and raw_result is not None:
            raw_cx, raw_cy = raw_result[0], raw_result[1]
            kf.x = np.array([[raw_cx], [raw_cy], [0.0], [0.0]], dtype=float)
            kf.P = np.eye(4, dtype=float) * INITIAL_P_AFTER_FIRST_MEAS
            initialized = True
        elif initialized:
            F = state_transition_F(dt)
            Q = process_noise_Q(dt)
            kf.predict(F=F, Q=Q)
            if raw_result is not None:
                raw_cx, raw_cy = raw_result[0], raw_result[1]
                z = np.array([[raw_cx], [raw_cy]], dtype=float)
                kf.update(z)

        if initialized:
            kf_state = (
                float(kf.x[0, 0]),
                float(kf.x[1, 0]),
                float(kf.x[2, 0]),
                float(kf.x[3, 0]),
            )
        else:
            kf_state = None

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
