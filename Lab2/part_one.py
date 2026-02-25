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
# ADC clock rate.. how quickly the ADC is updating (max of 30kHz)

# PRBS signal should be max at 15kHz... 
# is this the same thing as how fast we're toggling the drive line

# we're setting the drive lines up ourselves 

# sense_line = 
prbs_signal = bpsk(ninebit_lfsr())
notouch_signal = autocorrelation(prbs_signal,sense_line,511)

# find top five peaks and annotate each with (lag, value)
sorted_indices = np.argsort(notouch_signal)
top_n_indices = np.sort(sorted_indices[-5:])
for i in top_n_indices:
    lag, val = i, notouch_signal[i]