import os
import sys
import time
import atexit
import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0,
    os.path.join(
        _script_dir,
        "High-Precision-AD-DA-Board-Demo-Code",
        "RaspberryPI",
        "ADS1256",
        "python3",
    ),
)

import ADS1256
import PRBS_code_SG as prbs

# --- Hardware configuration ---
DRIVE = [7, 12, 16, 20, 21]          # BCM drive pins (lab requirement)
SENSE = [1, 2, 3, 4, 5, 6, 7]        # ADC sense channels

# --- Programmable Part 2 settings ---
N = 255                               # correlator/PRBS length (2^m - 1)
TAPS = prbs.taps_prbs255              # keep PRBS generator from PRBS_code_SG.py
PRBS_FREQ_HZ = 15000.0                # 15 kHz chip rate
N_SKIP = 5                            # skip initial samples after channel switch
N_AVG = 2                             # average captures per channel per frame
DRIVE_SENSE_LAG = 0                   # lag compensation (chips)

# ADS1256 config (same across all channels)
ADC_GAIN_KEY = "ADS1256_GAIN_32"
ADC_DRATE_KEY = "ADS1256_30000SPS"

# Saved plots
LIVE_PLOT_PATH = os.path.join(_script_dir, "check_live_plot_latest.png")
HEATMAP_PATH = os.path.join(_script_dir, "check_heatmap_latest.png")


def cleanup_gpio():
    for pin in DRIVE:
        GPIO.output(pin, 0)
    GPIO.cleanup(DRIVE)


def setup_hw():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in DRIVE:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    adc = ADS1256.ADS1256()
    adc.ADS1256_init()
    adc.ADS1256_WriteReg(0x00, 0x02)  # buffer ON (required)
    adc.ADS1256_ConfigADC(
        ADS1256.ADS1256_GAIN_E[ADC_GAIN_KEY],
        ADS1256.ADS1256_DRATE_E[ADC_DRATE_KEY],
    )
    return adc


def normalized_circular_corr(reference, signal):
    # Keep PRBS generation from PRBS_code_SG; use signed normalized correlation.
    ref = np.asarray(reference, dtype=float)
    sig = np.asarray(signal, dtype=float)
    ref = ref - np.mean(ref)
    sig = sig - np.mean(sig)
    denom = np.linalg.norm(ref) * np.linalg.norm(sig)
    if denom < 1e-12:
        return np.zeros(len(ref))

    out = np.zeros(len(ref), dtype=np.float64)
    for k in range(len(ref)):
        out[k] = np.sum(ref * np.roll(sig, k)) / denom
    return out


def setup_plot():
    plt.ion()
    fig, (ax_corr, ax_map) = plt.subplots(1, 2, figsize=(12, 5))
    lag_x = np.arange(N)

    lines = []
    for s_ch in SENSE:
        (line,) = ax_corr.plot(lag_x, np.zeros(N), label=f"Sense {s_ch}")
        lines.append(line)
    ax_corr.set_title("Correlator Outputs (All 7 Sense Channels)")
    ax_corr.set_xlabel("Lag")
    ax_corr.set_ylabel("Normalized correlation")
    ax_corr.set_xlim(0, N - 1)
    ax_corr.set_ylim(-1.0, 1.0)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend(loc="upper right", fontsize=8)

    heat = ax_map.imshow(
        np.zeros((len(SENSE), len(DRIVE))),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    ax_map.set_title("Baseline Capacitance Map (Sense x Drive)")
    ax_map.set_xlabel("Drive index")
    ax_map.set_ylabel("Sense channel")
    ax_map.set_xticks(np.arange(len(DRIVE)))
    ax_map.set_yticks(np.arange(len(SENSE)))
    ax_map.set_yticklabels([str(ch) for ch in SENSE])
    fig.colorbar(heat, ax=ax_map, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax_corr, ax_map, lines, heat


def main():
    if N <= 0 or (N + 1) & N:
        raise ValueError("N must be of form 2^m - 1 (e.g., 31,127,255,511,1023)")

    # PRBS generation from your PRBS_code_SG.py
    seq0 = prbs.prbs_generator(N, TAPS, phase=0)
    shift = N // 5
    seq1 = prbs.prbs_generator(N, TAPS, phase=1 * shift)
    seq2 = prbs.prbs_generator(N, TAPS, phase=2 * shift)
    seq3 = prbs.prbs_generator(N, TAPS, phase=3 * shift)
    seq4 = prbs.prbs_generator(N, TAPS, phase=4 * shift)
    prbs_matrix = np.vstack([seq0, seq1, seq2, seq3, seq4])
    ref_bpsk = 2 * seq0 - 1

    phase_idx = [((k * shift) + DRIVE_SENSE_LAG) % N for k in range(5)]
    dt = 1.0 / PRBS_FREQ_HZ

    adc = setup_hw()
    atexit.register(cleanup_gpio)

    fig, ax_corr, ax_map, lines, heat = setup_plot()
    fig.savefig(LIVE_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saving live plots to: {LIVE_PLOT_PATH} and {HEATMAP_PATH}")

    frame_count = 0
    t0 = time.perf_counter()
    t_prev = t0

    while True:
        xcor_raw = np.zeros((len(SENSE), N), dtype=np.float64)
        xcor_map = np.zeros((len(SENSE), len(DRIVE)), dtype=np.float64)

        # Sequential sense scan with PRBS restart after each channel switch.
        for j, sense_ch in enumerate(SENSE):
            corr_acc = np.zeros(N, dtype=np.float64)
            for _ in range(N_AVG):
                temp = []
                total = N + N_SKIP
                for i in range(total):
                    loop_start = time.perf_counter()
                    s = i - N_SKIP if i >= N_SKIP else i

                    for d_i, pin in enumerate(DRIVE):
                        GPIO.output(pin, int(prbs_matrix[d_i, s]))

                    raw = adc.ADS1256_GetChannalValue(sense_ch)
                    temp.append(raw * 5.0 / 0x7FFFFF)

                    while time.perf_counter() - loop_start < dt:
                        pass

                sig = np.array(temp[N_SKIP:], dtype=np.float64)
                corr_acc += normalized_circular_corr(ref_bpsk, sig)

            xcor_raw[j, :] = corr_acc / N_AVG
            xcor_map[j, :] = [xcor_raw[j, idx] for idx in phase_idx]

        for j, line in enumerate(lines):
            line.set_ydata(xcor_raw[j, :])

        peak = max(1e-6, np.max(np.abs(xcor_raw)))
        ax_corr.set_ylim(-min(1.0, 1.1 * peak), min(1.0, 1.1 * peak))

        heat.set_data(xcor_map)
        map_peak = max(1e-6, np.max(np.abs(xcor_map)))
        heat.set_clim(-map_peak, map_peak)

        frame_count += 1
        now = time.perf_counter()
        inst_fps = 1.0 / max(now - t_prev, 1e-6)
        avg_fps = frame_count / max(now - t0, 1e-6)
        t_prev = now

        ax_corr.set_title(
            f"Correlator Outputs | frame {frame_count} | inst {inst_fps:.2f} Hz | avg {avg_fps:.2f} Hz"
        )
        fig.canvas.draw_idle()
        plt.pause(0.001)

        # Save latest snapshots for debugging/logging.
        fig.savefig(LIVE_PLOT_PATH, dpi=150, bbox_inches="tight")
        extent = ax_map.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(HEATMAP_PATH, dpi=150, bbox_inches=extent.expanded(1.05, 1.15))

        if frame_count % 10 == 0:
            print(
                f"Frame {frame_count}: inst_fps={inst_fps:.2f}, avg_fps={avg_fps:.2f}, "
                f"N={N}, phase_idx={phase_idx}, lag={DRIVE_SENSE_LAG}"
            )


if __name__ == "__main__":
    main()