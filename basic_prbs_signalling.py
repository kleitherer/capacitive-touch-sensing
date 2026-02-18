"""
Pseudo-Random Binary Sequence Signalling

Used for differentiating between signals, good for privacy (GPS ranging)


"""
from PRBS_utils import *

import numpy as np
import matplotlib.pyplot as plt

"""
Part B: plot autocorrelated signal of PRBS511
"""
prbs_signal_unmapped = ninebit_lfsr()
prbs_signal = bpsk(prbs_signal_unmapped)
autocorrelated_signal = autocorrelation(prbs_signal, prbs_signal, 511)
# confirm that max should be at index = 0
print("Index of max of autocorrelated signal:", np.argmax(autocorrelated_signal))
plt.plot(autocorrelated_signal)
plt.annotate(
    'Spike with height 511 at zero offset',
    xy=(0, 511), 
    xytext=(10, 470),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=7
)
plt.grid()
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation curve for PRBS511")
plt.show()


"""
Part B: plot subsequence of length 255 from PRBS511 with smallest off peak autocorrelation
"""
max_index = 0
max_value = float('inf')
max_values = []

prbs_signal = bpsk(prbs_signal_unmapped)
prbs = ninebit_lfsr()
for i in range(0, 257): 
    prbs_cut = np.array(prbs[i:255 + i])
    prbs_cut_bpsk = bpsk(prbs_cut)
    autocorr_subsequence = autocorrelation(prbs_cut_bpsk, prbs_cut_bpsk, 254)
    off_peak_max = np.max(autocorr_subsequence[1:]) if len(autocorr_subsequence) > 1 else 0
    if off_peak_max < max_value:
        max_index = i
        max_value = off_peak_max
        max_values.append(max_value)
print("Index of subsequence with smallest max off-peak correlation:", max_index)
print("Smallest maximum off-peak correlation:", max_value)
prbs_cut = np.array(prbs[max_index:255 + max_index])
prbs_cut_bpsk = bpsk(prbs_cut)
autocorr_subsequence = autocorrelation(prbs_cut_bpsk, prbs_cut_bpsk, 254)
plt.plot(autocorr_subsequence)
plt.grid()
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title(f"Best 255-bit Subsequence (smallest maximum off-peak correlation ({max_value}))")
plt.show()


"""
Part C: extra credit, compare PRBS sequences with same length but different polynomials
Call the PRBS511 sequence you generated in 3a as seqA, and this new generator
polynomial seqB. Is seqA equal to seqB? What is the maximal value of the cross-
correlation between seqA and seqB?

the maximal value is 511 (shouldn't it be 512), so it means that they are the same and they are equal

Cross-correlation measures the similarity between two different signals (or time series) as 
a function of the lag (displacement) between them, while autocorrelation measures the similarity 
of a single signal with a lagged version of itself

"""
extra_credit_prbs_signal_unmapped = extracredit_ninebit_lfsr()
#print(len(extra_credit_prbs_signal_unmapped))
extra_credit_prbs_signal = bpsk(extra_credit_prbs_signal_unmapped)
autocorr_sequence = autocorrelation(prbs_signal, extra_credit_prbs_signal, 511)
print("Length of autocorrelated subsequence: ", len(autocorr_sequence))
print("Max value of cross-correlation between seqA and seqB: ", np.max(autocorr_sequence))
print("Index of max value ", np.argmax(autocorr_subsequence))
plt.plot(autocorr_sequence)
plt.grid()
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation curve for seqA and seqB")
plt.show()