import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO


_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256


DRIVE_PINS = [7, 12, 16, 20, 21]          
SENSE_CHANNELS = [1, 2, 3, 4, 5, 6, 7]    

PRBS_ORDER = 8                             # change to be btwn 2...16
PRBS_SEED = 0x01                           # initial phase everything is shifted from this cant be 0
PRBS_TAPS_MASK = 0xB8                      # taps mask for PRBS8
PRBS_FREQ_HZ = 15000.0                     # 15 kHz 

N_SKIP = 5                                 # samples skipped after channel switch
N_AVG = 2                                  # frame averaging per channel

ADC_GAIN_KEY = "ADS1256_GAIN_32"
ADC_DRATE_KEY = "ADS1256_30000SPS"

# if drive/sense timing is offset, tune this.
DRIVE_SENSE_LAG = 0

# Plot saving
LIVE_PLOT_PATH = os.path.join(_script_dir, "part2_live_plot_latest.png")
HEATMAP_PATH = os.path.join(_script_dir, "part2_heatmap_latest.png")
SAVE_EVERY_N_FRAMES = 1


def lfsr_prbs(order: int, taps_mask: int, seed: int) -> np.ndarray:
    n = (1 << order) - 1
    state = seed & ((1 << order) - 1)

    seq = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        seq[i] = state & 0x1
        tapped = state & taps_mask
        parity = 0
        while tapped:
            parity ^= (tapped & 0x1)
            tapped >>= 1
        state = (state >> 1) | (parity << (order - 1))
        state &= (1 << order) - 1
    return seq


def build_prbs_matrix(base_seq: np.ndarray) -> np.ndarray:
    n = len(base_seq)
    shift = n // 5
    return np.vstack([
        np.roll(base_seq, 0 * shift),
        np.roll(base_seq, 1 * shift),
        np.roll(base_seq, 2 * shift),
        np.roll(base_seq, 3 * shift),
        np.roll(base_seq, 4 * shift),
    ])


def normalized_circular_corr(ref: np.ndarray, sig: np.ndarray) -> np.ndarray:
    ref = ref.astype(float) - np.mean(ref)
    sig = sig.astype(float) - np.mean(sig)
    denom = np.linalg.norm(ref) * np.linalg.norm(sig)
    if denom < 1e-12:
        return np.zeros(len(ref))

    out = np.zeros(len(ref), dtype=np.float64)
    for k in range(len(ref)):
        out[k] = np.sum(ref * np.roll(sig, k)) / denom
    return out


def setup_hw():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in DRIVE_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    adc = ADS1256.ADS1256()
    adc.ADS1256_init()
    adc.ADS1256_WriteReg(0x00, 0x02)  # buffer ON
    adc.ADS1256_ConfigADC(
        ADS1256.ADS1256_GAIN_E[ADC_GAIN_KEY],
        ADS1256.ADS1256_DRATE_E[ADC_DRATE_KEY],
    )
    return adc


def scan_one_frame(adc, prbs_matrix: np.ndarray, ref_bpsk: np.ndarray):
    n = prbs_matrix.shape[1]
    dt = 1.0 / PRBS_FREQ_HZ
    shift = n // 5
    phase_idx = [((k * shift) + DRIVE_SENSE_LAG) % n for k in range(5)]

    raw_sense = np.zeros((len(SENSE_CHANNELS), n), dtype=np.float64)
    xcor_raw = np.zeros((len(SENSE_CHANNELS), n), dtype=np.float64)
    xcor_map = np.zeros((len(SENSE_CHANNELS), len(DRIVE_PINS)), dtype=np.float64)

    for j, ch in enumerate(SENSE_CHANNELS):
        adc_acc = np.zeros(n, dtype=np.float64)
        corr_acc = np.zeros(n, dtype=np.float64)

        # Restart PRBS from s=0 for each sense channel (required behavior).
        for _ in range(N_AVG):
            temp = []
            total = n + N_SKIP
            for i in range(total):
                t0 = time.perf_counter()
                s = (i - N_SKIP) if i >= N_SKIP else i

                for p_i, pin in enumerate(DRIVE_PINS):
                    GPIO.output(pin, int(prbs_matrix[p_i, s]))

                raw = adc.ADS1256_GetChannalValue(ch)
                temp.append(raw * 5.0 / 0x7FFFFF)

                while time.perf_counter() - t0 < dt:
                    pass

            sig = np.array(temp[N_SKIP:], dtype=np.float64)
            adc_acc += sig
            corr_acc += normalized_circular_corr(ref_bpsk, sig)

        raw_sense[j, :] = adc_acc / N_AVG
        xcor_raw[j, :] = corr_acc / N_AVG
        xcor_map[j, :] = [xcor_raw[j, idx] for idx in phase_idx]

    return raw_sense, xcor_raw, xcor_map, phase_idx


def setup_realtime_plot(n_lags: int):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    lag_x = np.arange(n_lags)
    lines = []
    for ch in SENSE_CHANNELS:
        (line,) = ax1.plot(lag_x, np.zeros(n_lags), label=f"Ch{ch}", linewidth=1.0)
        lines.append(line)
    ax1.set_title("Correlator Outputs vs Lag")
    ax1.set_xlabel("Lag (chips)")
    ax1.set_ylabel("Normalized correlation")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_lags - 1)
    ax1.set_ylim(-1.0, 1.0)
    ax1.legend(loc="upper right", fontsize=8)

    heat = ax2.imshow(
        np.zeros((len(SENSE_CHANNELS), len(DRIVE_PINS))),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    ax2.set_title("Sense x Drive Correlation Map")
    ax2.set_xlabel("Drive index")
    ax2.set_ylabel("Sense channel")
    ax2.set_xticks(np.arange(len(DRIVE_PINS)))
    ax2.set_yticks(np.arange(len(SENSE_CHANNELS)))
    ax2.set_yticklabels([str(ch) for ch in SENSE_CHANNELS])
    fig.colorbar(heat, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw()
    return fig, ax1, ax2, lines, heat


if __name__ == "__main__":
    base_prbs = lfsr_prbs(PRBS_ORDER, PRBS_TAPS_MASK, PRBS_SEED)
    n = len(base_prbs)
    ref_bpsk = 2 * base_prbs - 1
    prbs_matrix = build_prbs_matrix(base_prbs)

    print("=== Part 2 realtime correlator ===")
    print(f"PRBS order={PRBS_ORDER}, length={n}")
    print(f"PRBS freq={PRBS_FREQ_HZ} Hz, ADC rate={ADC_DRATE_KEY}, gain={ADC_GAIN_KEY}")
    print(f"N_SKIP={N_SKIP}, N_AVG={N_AVG}, lag compensation={DRIVE_SENSE_LAG}")
    print(f"Will save full plot to: {LIVE_PLOT_PATH}")
    print(f"Will save heatmap to:   {HEATMAP_PATH}")

    adc = setup_hw()

    fig, ax1, ax2, lines, heat = setup_realtime_plot(n)
    fig.canvas.draw()
    fig.savefig(LIVE_PLOT_PATH, dpi=150, bbox_inches="tight")
    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(HEATMAP_PATH, dpi=150, bbox_inches=extent.expanded(1.05, 1.15))
    print("Saved initial plot snapshots.")

    frame_count = 0
    t_start = time.perf_counter()
    t_prev = time.perf_counter()

    while True:
        _, xcor_raw, xcor_map, phase_idx = scan_one_frame(adc, prbs_matrix, ref_bpsk)

        for j in range(len(SENSE_CHANNELS)):
            lines[j].set_ydata(xcor_raw[j, :])

        peak = max(1e-6, np.max(np.abs(xcor_raw)))
        ax1.set_ylim(-1.0, 1.0)

        heat.set_data(xcor_map)
        hpeak = max(1e-6, np.max(np.abs(xcor_map)))
        heat.set_clim(-hpeak, hpeak)

        frame_count += 1
        now = time.perf_counter()
        inst_fps = 1.0 / max(now - t_prev, 1e-6)
        avg_fps = frame_count / max(now - t_start, 1e-6)
        t_prev = now

        ax1.set_title(f"Correlator Outputs vs Lag | frame {frame_count} | inst {inst_fps:.2f} Hz | avg {avg_fps:.2f} Hz")
        fig.canvas.draw()
        plt.pause(0.001)

        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            # Save full figure (correlators + heatmap)
            try:
                fig.canvas.draw()
                fig.savefig(LIVE_PLOT_PATH, dpi=150, bbox_inches="tight")
                # Save heatmap-only image for quick viewing
                extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(HEATMAP_PATH, dpi=150, bbox_inches=extent.expanded(1.05, 1.15))
            except Exception as e:
                print(f"Plot save error: {e}")

        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: inst_fps={inst_fps:.2f}, avg_fps={avg_fps:.2f}, phase_idx={phase_idx}")
            print(f"Saved plots: {LIVE_PLOT_PATH} | {HEATMAP_PATH}")
