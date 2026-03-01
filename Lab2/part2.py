import os
import sys
import threading
import socket

import time
import numpy as np

# WEB_VIEWER: open http://<pi-ip>:8080 in your browser for live view
# SAVE_ONLY: just save to file (no web server)
WEB_VIEWER = True
WEB_PORT = 8080
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
    """Set all DRIVE GPIO pins for PRBS index s."""
    for i, pin in enumerate(DRIVE):
        GPIO.output(pin, int(prbs_matrix[i, s]))

def _get_local_ip():
    """Get Pi's IP for display in browser URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def _run_web_server(port, serve_dir):
    """Serve heatmap image + auto-refresh HTML on port."""
    import http.server
    os.chdir(serve_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("", port), handler) as httpd:
        httpd.serve_forever()

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

    # ---- Plot setup ----
    print("Setting up the plot")
    fig, ax = plt.subplots(1, 1)
    heatmap = ax.imshow(xcor, cmap='viridis', interpolation='nearest', vmin=threshold)
    plt.colorbar(heatmap)
    ax.invert_yaxis()
    ax.invert_xaxis()
    plt.title("Heatmap (Sense x Drive)")

    if WEB_VIEWER:
        # Create HTML that auto-refreshes the image
        lab_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(lab_dir, "heatmap_viewer.html")
        with open(html_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html><head><title>Touch Sensor Live</title></head>
<body style="margin:20px;background:#1a1a1a;color:#eee;">
<h2>Touch Sensor Heatmap (live)</h2>
<p>Refreshes every 0.5 s</p>
<img src="heatmap_latest.png" id="img" style="max-width:90%;" onerror="this.style.display='none'">
<script>
setInterval(function(){
  var i = document.getElementById('img');
  i.src = 'heatmap_latest.png?t=' + Date.now();
}, 500);
</script>
</body></html>""")
        server = threading.Thread(target=_run_web_server, args=(WEB_PORT, lab_dir), daemon=True)
        server.start()
        pi_ip = _get_local_ip()
        print(f"Live view: open http://{pi_ip}:{WEB_PORT}/heatmap_viewer.html in your browser")
    print("Plot setup successful")

    # ---- Timing targets (optional) ----
    # Your original code sets a target PRBS bit rate; in practice Python/GPIO/ADS1256 calls dominate.
    target_freq = 15e3
    target_period = 1.0 / target_freq

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
                loop_start = time.time()

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

        # Threshold and update plot ONCE per frame (after all sense lines scanned)
        xcor_plot = xcor.copy()
        xcor_plot[xcor_plot < threshold] = 0

        heatmap.set_data(xcor_plot)
        fig.savefig(SAVE_PATH, dpi=150)

        frame_count += 1

        # Optional: print frame rate occasionally
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frames: {frame_count}, approx frame rate: {fps:.3f} Hz")