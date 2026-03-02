import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import ADS1256
import RPi.GPIO as GPIO
import spidev
def generate_prbs(polynomial, length, seed):
"""Generate a Pseudo-Random Bit Sequence (PRBS) based on a given polynomial and
Args:
polynomial (int): Polynomial used for the feedback in the LFSR (e.g., 0b110
length (int): The length of the PRBS sequence (e.g., 127 for PRBS7).
seed (int): Initial seed (non-zero) for the LFSR.
Returns:
list: List containing the PRBS sequence as binary bits.
"""
num_bits = polynomial.bit_length()
lfsr = [int(bit) for bit in format(seed, f'0{num_bits}b')]
prbs_sequence = []
for _ in range(length):
# Output the last bit in the register (can be any bit depending on the desi
prbs_sequence.append(lfsr[-1])
# print(lfsr, prbs_sequence[-1])
# XOR the tapped bits to calculate feedback bit
feedback = 0
for bit_position in range(num_bits):
# print("pos: ", bit_position)
if (polynomial >> bit_position) & 1: # Check if this position has a ta
feedback ^= lfsr[-(bit_position + 1)]
# Shift the register to the right and insert the feedback bit at the front
lfsr = lfsr[1:] + [feedback]
return prbs_sequence
def circular_cross_correlation(x, y):
"""Compute the circular cross-correlation of two 1D signals of arbitrary length
Args:
x (np.ndarray): First input signal array.
y (np.ndarray): Second input signal array.
Returns:
np.ndarray: Circular cross-correlation result.
"""
# Determine the length of the result, which will match the length of the longer
N = max(len(x), len(y))
# Zero-pad both sequences to the same length
x = np.pad(x, (0, N - len(x)), mode='constant')
y = np.pad(y, (0, N - len(y)), mode='constant')
result = np.zeros(N)
# Compute circular cross-correlation
for k in range(N):
result[k] = np.sum(x * np.roll(y, k))
return result
# Drive and sense line pins as wired
DRIVE = [21, 7, 12, 16, 20]
SENSE = [1, 2, 3, 4, 5, 6, 7]
SETUP_SPI = True
if __name__ == '__main__':
ADC = ADS1256.ADS1256()
ADC.ADS1256_init()
# initialization
ADDR = 0x00
ON = 0x02
ADC.ADS1256_WriteReg(ADDR, ON)
# SPI setup
if SETUP_SPI:
SPI = spidev.SpiDev(0, 0)
SPI.mode = 0b01
SPI.max_speed_hz = 3000000
# GPIO setup
# GPIO.setmode(GPIO.BOARD)
for drive in DRIVE:
GPIO.setup(drive, GPIO.OUT)
polynomial = 0x60 # 0x14
length = 2**7-1 # 5
# polynomial = 0x06
# length = 2**7-1
shift = length // 5
seed = 0x01
prbs0 = generate_prbs(polynomial, length, seed)
prbs1 = np.roll(prbs0, 1*shift)
prbs2 = np.roll(prbs0, 2*shift)
prbs3 = np.roll(prbs0, 3*shift)
prbs4 = np.roll(prbs0, 4*shift)
prbs = np.vstack([prbs0, prbs1, prbs2, prbs3, prbs4])
raw_sense = np.zeros((7, length))
# Record the start time
start_time = time.time()
# Target frequency and interval
target_freq = 15 * 10**3 # kHz
target_period = 1.0 / target_freq # Interval in seconds
s = 0

xcor = np.zeros((np.shape(raw_sense)[0], len(DRIVE)))
xcor_raw = np.zeros((np.shape(raw_sense)[0], len(prbs0)))
sense = np.zeros((7, length))
threshold = 100
# Create a figure and axis for the heatmap
figure, ax = plt.subplots(1, 1)
heatmap = ax.imshow(xcor, cmap='viridis', interpolation='nearest', vmin=thresho
plt.colorbar(heatmap)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Live Updating Heatmap")
# Draw and cache the background
figure.canvas.draw()
axbackground = figure.canvas.copy_from_bbox(ax.bbox)
plt.show(block=False)
print("Plot setup successful")
count = 0
while True:
# One by one
for j in range(len(SENSE)):
# Set GPIO outputs based on PRBS
for i in range(len(DRIVE)):
GPIO.output(DRIVE[i], int(prbs[i, s]))
ADC.ADS1256_SetChannal(SENSE[j])
ADC_Value = ADC.ADS1256_GetChannalValue(SENSE[j])
loop_start_time = time.time() # Record the time when the loop starts

raw_sense[j][s] = ADC_Value * 5.0 / 0x7fffff
s = s + 1
if (s >= length):
for idx in range(np.shape(raw_sense)[0]):
xcor_raw[idx] = circular_cross_correlation(prbs0, raw_sense[idx])
xcor[idx] = [xcor_raw[idx][0], xcor_raw[idx][1*shift], xcor_raw[idx

# Update the heatmap data with new random values (replace this with you
touch_map = xcor
touch_map[touch_map < threshold] = 0
heatmap.set_data(touch_map)
# Restore background and redraw the heatmap
figure.canvas.restore_region(axbackground)
ax.draw_artist(heatmap)
figure.canvas.blit(ax.bbox)
figure.canvas.flush_events()
# print("pausing")
# # Pause briefly to allow the GUI to update
# plt.pause(0.001) # A short pause to ensure the plot updates smoothly
count += 1
s = 0
# # FPS calculator
# if count > 10:
# current_time = time.time()
# print("Frame Rate: ", count / (current_time - start_time))
# else:
# elapsed_time = time.time() - start_time
# # Calculate loop duration
# loop_duration = time.time() - loop_start_time
# # Sleep for the remaining time to maintain 15KHz rate
# if loop_duration < target_period:
# time.sleep(target_period - loop_duration)