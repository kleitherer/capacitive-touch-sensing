"""
PRBS (Pseudo-Random Binary Sequence) generator and correlation utilities.
Converted from PRBS_code_SG.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Tap constants for various PRBS lengths
taps_prbs7 = 0x6
taps_prbs15 = 0xC
taps_prbs127 = 0x60
taps_prbs255 = 0x8E   # PRBS-8: length 2^8-1 = 255
taps_prbs511 = 0x110
taps_prbs1023 = 0x240


def prbs_generator(N, taps, initial_value=None, phase=0):
    """Generic PRBS generator using LFSR."""
    output = np.zeros(N)
    nbits = int(np.log2((N + 1)))  # N = 2^nbits - 1
    print(f"Number of bits: {nbits}")

    if initial_value is not None:
        seq_values = initial_value
    else:
        seq_values = np.ones(nbits, dtype=int)

    print(f"Initial sequence: {seq_values}")

    taps_array = np.array([int(bit) for bit in format(taps, f'0{nbits}b')])
    print(f"Taps array: {taps_array}")
    ones_indices = np.where(taps_array == 1)[0]
    print(f"Ones indices: {ones_indices}")

    n_taps = len(ones_indices)
    xor_indices = [nbits - 1 - ones_indices[i] for i in range(n_taps)]
    print(f"Xor indices: {xor_indices}")

    for i in range(N):
        output[i] = seq_values[-1]
        vals_to_xor = seq_values[xor_indices]
        xor_result = np.bitwise_xor.reduce(vals_to_xor)
        seq_values = np.roll(seq_values, 1)
        seq_values[0] = xor_result

    if phase != 0:
        output = np.roll(output, phase)

    print(f"Final output: {output}")
    return output


def corr(array_1, array_2, N, plot='n', title=None):
    """Correlation of two arrays. Returns correlation array of length N."""
    corr_array = [0] * N
    for n in range(0, N):
        corr_array[n] = np.abs((np.sum(array_1 * (np.roll(array_2, n)))))

    if plot == 'y':
        plt.plot(corr_array)
        if title is not None:
            plt.title(title)
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.show()

    return corr_array


if __name__ == "__main__":
    # Demo: generate PRBS511
    prbs511 = prbs_generator(511, taps_prbs511)

    # Demo: phase-shifted PRBS511
    prbs511_phase = prbs_generator(511, taps_prbs511, phase=10)

    # Demo: autocorrelation of PRBS511 (Question 1 part b)
    prbs_511_output = prbs_generator(511, taps_prbs511)
    prbs_511_mapped = 2 * prbs_511_output - 1  # mapped to -1/+1
    prbs_511_autocorr = corr(
        prbs_511_mapped, prbs_511_mapped, 511,
        plot='y', title='Autocorrelation of PRBS 511'
    )
