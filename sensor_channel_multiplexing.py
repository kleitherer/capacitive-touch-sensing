"""
Sensor Channel Multiplexing 
Our sense and drive lines of the capacitive touch display is driven with pseudo-random binary sequences such that
noisy signals can be more easily sensed (using autocorrelation). We're able to separately measure the signal from 
each drive line into the sense line.

It also allows us to speed up the touch sensors because by being able to differentiate waveforms on the same sense 
row from each other, we can drive all lines in parallel.

Givens: ADC waveform output of sense line from HW3.Pr.notouch.txt
- here each drive line has a different phase of the same modulated PRBS511 signal

To do: correlate against PRBS511 sequence to find the waveform for the 5 correlation peaks


the sense line is the superposition of all the drive lines


there are 5 correlation peaks (each peak will have a correlator output which is the mutual drive/sense capacitance for each of 5 drive lines)

each peak will have a phase offset...

"""
import numpy as np
import matplotlib.pyplot as plt
from PRBS_utils import *
from scipy.optimize import curve_fit

# read the txt file into an array... it is the superposition of all the drive lines
sense_line = np.loadtxt('HW3.Pr3.notouch.txt')
for i in range(0,len(sense_line)):
    sense_line[i] = sense_line[i].item()

prbs_signal = bpsk(ninebit_lfsr())
notouch_signal = autocorrelation(prbs_signal,sense_line,511)
plt.plot(notouch_signal)

# find top five peaks and annotate each with (lag, value)
sorted_indices = np.argsort(notouch_signal)
top_n_indices = np.sort(sorted_indices[-5:])
for i in top_n_indices:
    lag, val = i, notouch_signal[i]
    plt.annotate(
        f'(Lag: {lag}, Val:{val:.0f})',
        xy=(lag, val),
        xytext=(lag + 15, val),
        fontsize=7,
        arrowprops=dict(facecolor='black', arrowstyle='->'),
    )
plt.grid()
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation between no touch ADC waveform and PRBS511")
plt.show()
sorted_indices = np.argsort(notouch_signal)
top_n_indices = np.sort(sorted_indices[-5:])
for i in top_n_indices:
    print(f"At lag {i}, relative capacitance is", notouch_signal[i])


print("\n ****** \n")
"""
b) (10 points) In the even-more-creatively named file HW3.Pr3.touch.txt, there is a touch signal
modulating the capacitances. Using the HW3.Pr3.notouch.txt waveform as the notouch baseline
of the sensor, where is the touch located? The physical coordinates (in millimeters) of the
drive/sense pairs is given below. 

The signal with the smallest delay corresponds to the drive/sense pair on the left (5 mm), 
the signal with the second smallest corresponds to 10 mm etc. You can assume the finger 
sensor response is a Gaussian in space.
"""

# plot the cross-correlation
sense_line_2 = np.loadtxt('HW3.Pr3.touch.txt')
for i in range(0,len(sense_line_2)):
    sense_line_2[i] = sense_line_2[i].item()

touch_signal = autocorrelation(prbs_signal,sense_line_2,511)
top_n_indices_2 = np.sort(np.argsort(touch_signal)[-5:])
change_in_cap = np.empty(len(touch_signal))
for idx, i in enumerate(top_n_indices_2):
    lag, val = i, touch_signal[i]
    plt.annotate(
        f'(Lag: {lag}, Val:{val:.0f})',
        xy=(lag, val),
        xytext=(lag + 15, val),
        fontsize=7,
        arrowprops=dict(facecolor='black', arrowstyle='->'),
    )
    print(f"At lag {i}, relative cap is", touch_signal[i])
    change_in_cap[i] = notouch_signal[i] - touch_signal[i]
plt.plot(touch_signal)
plt.grid()
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation between touch ADC waveform and PRBS511 sequence")
plt.show()

# now we need to fit these to a gaussian (in space: finger response is Gaussian in mm)
# smallest delay = 5 mm, second = 10 mm, ... so positions are 5, 10, 15, 20, 25 mm
x_mm = np.array([5., 10., 15., 20., 25.])
y_cap = np.array([touch_signal[i] - notouch_signal[i] for i in top_n_indices_2])

def gaussian(x, amplitude, mu, std_dev):
    return amplitude * np.exp(-(x - mu)**2 / (2 * std_dev**2))

initial_guess = [np.max(y_cap), np.mean(x_mm), 5.0]

optimal_params, cov_matrix = curve_fit(gaussian, x_mm, y_cap, p0=initial_guess)
fit_amplitude, fit_mean, fit_std_dev = optimal_params
x_smooth = np.linspace(x_mm.min(), x_mm.max(), 200)

plt.scatter(x_mm, y_cap, label='Change in capacitance at 5 sense positions')
plt.plot(x_smooth, gaussian(x_smooth, *optimal_params), color='red', linewidth=2, label='Fitted Gaussian')
plt.title("Gaussian fit: touch location in space")
plt.xlabel("Position (mm)")
plt.ylabel("Change in capacitance")
plt.axvline(fit_mean, color='g', linestyle='--', linewidth=1)
plt.legend()
plt.grid()
plt.show()

print(f"Touch center (fitted mean): {fit_mean:.2f} mm")
print(f"Fitted standard deviation: {fit_std_dev:.2f} mm")
print(f"Fitted amplitude: {fit_amplitude:.2f}")

"""
c) Extra Credit: How much noise is there in each sample in the HW3.Pr3.notouch.txt? 
before the correlation is done?


"""

prbs = bpsk(ninebit_lfsr()) 
corr = autocorrelation(prbs, sense_line, 511)

peak_indices = np.sort(np.argsort(corr)[-5:])

relative_amplitudes = np.array([corr[i] for i in peak_indices])

relative_amplitudes /= np.max(relative_amplitudes)


reconstructed = np.zeros_like(sense_line)

for amp, lag in zip(relative_amplitudes, peak_indices):
    shifted_prbs = np.roll(prbs, lag)
    reconstructed += amp * shifted_prbs

# --- Find best global scaling factor A (least squares fit) ---

A = np.dot(sense_line, reconstructed) / np.dot(reconstructed, reconstructed)

estimated_signal = A * reconstructed


noise = sense_line - estimated_signal


noise_mean = np.mean(noise)
noise_std = np.std(noise)
noise_rms = np.sqrt(np.mean(noise**2))

print("Estimated drive amplitude scaling A:", A)
print("Noise mean:", noise_mean)
print("Noise std dev:", noise_std)
print("Noise RMS:", noise_rms)

plt.figure()
plt.plot(noise)
plt.title("Noise (No-Touch)")
plt.xlabel("Sample index")
plt.ylabel("amplitude")
plt.grid()
plt.show()