"""
Write (and submit) the appropriate Python code to drive out a PRBS signal on 5 GPIO lines
coming out of the RPi4. 

The PRBS length should be flexible, since you will be experimenting
with different lengths (up to a 16-bit shift register, or a sequence length of 65535). 

All 5 drive lines should use the same PRBS code, with each drive using a different phase of the code
(this ensures orthogonality of the signals from each drive line). Since the ADC is clocked at
30kHz, the PRBS signal should be clocked at 15kHz to not require timing recovery. 

Keep in mind, the drive clock and the ADC sense clock are not phase locked in this system, so there
will be timing drift. Depending on how well you optimize your code the actual sampling rate of
the ADC may be lower (>20 kHz). You can also try to clock the PRBS at the same rate as
the ADC if you find the timing drift is small.
"""

from PRBS_utils import *
import matplotlib.pyplot as plt
import time
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "High-Precision-AD-DA-Board-Demo-Code", "RaspberryPI", "ADS1256", "python3"))
import ADS1256
import spidev
import RPi.GPIO as GPIO

# we're setting the drive lines up ourselves ... so we have to use readADCdata
ADC = ADS1256.ADS1256()
ADC.ADS1256_init()
# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21, GPIO.OUT) # drive 5 (pin 4)
GPIO.setup(20, GPIO.OUT) # drive 4 (pin 3)
GPIO.setup(16, GPIO.OUT) # drive 3 (pin 2)
GPIO.setup(12, GPIO.OUT) # drive 2 (pin 1)
GPIO.setup(7, GPIO.OUT) # drive 1 (pin 0)






# we need to iterate through each channel...
# do we do getChannalValue or readADCvalue

adc_value = ADC.ADS1256_GetChannalValue(7)*5.0/0x7fffff # this should be clocked at 30kHz


# we're driving the 5 GPIO pins and each PRBS signal is associated with a delay since each drive line has its own delay
# it comes from the GPIO pins ... do we need to initialize them? each one
prbs_signal = bpsk(lfsr(...))
GPIO.output(21, prbs_signal)

touch_signal = autocorrelation(prbs_signal,adc_value,511)





sorted_indices = np.argsort(touch_signal)
top_n_indices = np.sort(sorted_indices[-5:])
for i in top_n_indices:
    lag, val = i, touch_signal[i]


plt.grid()
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation between no touch ADC waveform and PRBS511")
plt.show()