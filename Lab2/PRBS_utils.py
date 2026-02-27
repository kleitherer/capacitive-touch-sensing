"""
Functions for PRBS CDMA
"""
import numpy as np
# PRBS7
def threebit_lfsr():
    mls_codeword = []
    a = [1] * 3
    chip = 0
    while chip < 7:
        mls_codeword.append(a[0])
        left_bit = a[0] ^ a[1]
        a = a[1:] + [left_bit]
        chip += 1
    return mls_codeword

# PRBS127
def sevenbit_lfsr():
    mls_codeword = []
    a = [1] * 7
    chip = 0
    while chip < 127:
        mls_codeword.append(a[0])
        left_bit = a[0] ^ a[1]
        a = a[1:] + [left_bit]
        chip += 1
    return mls_codeword


# PRBS511 with taps as 100010000
def ninebit_lfsr():
    mls_codeword = []
    #a = [1] * 9
    a = [1,0,0,0,0,0,0,0,1]
    chip = 0
    while chip < 511:
        mls_codeword.append(a[-1])
        right_bit = a[8] ^ a[4]
        a = [right_bit] + a[:-1] 
        chip += 1
    return mls_codeword
# Extra credit: Another possible generator polynomial for a PRBS511 sequence is given as: (polynomial is now 0x108)
# 100001000
def extracredit_ninebit_lfsr():
    mls_codeword = []
    a = [1,0,0,0,0,0,0,0,1]
    chip = 0
    while chip < 511:
        mls_codeword.append(a[0])
        right_bit = a[8] ^ a[3]
        a = [right_bit] + a[:-1] 
        chip += 1
    return mls_codeword

# PRBS1023
def tenbit_lfsr():
    """
    Generates a 10-bit LFSR m-sequence of length 1023 using taps at bits 3 and 10.
    """
    mls_codeword = []
    a = [1] * 10
    chip = 0
    while chip < 1023:
        mls_codeword.append(a[0])
        left_bit = a[0] ^ a[3]
        a = a[1:] + [left_bit]
        chip += 1
    return mls_codeword

def bpsk(codeword):
    """
    Applies BPSK modulation to a binary codeword:
    - Maps 0 to 1
    - Maps 1 to -1
    """
    codeword = np.array(codeword)
    return np.where(codeword == 0, 1, -1)

def autocorrelation(codeword_one, codeword_two, N):
    """
    Return an array of length N where each element holds the normalized dot product of the 
    time-delayed signal and the original signal
    """
    corr_array = [0]*N
    for n in range(0,N):
        corr_array[n] = np.abs((np.sum(codeword_one * (np.roll(codeword_two,-n)))))
    return corr_array

# def random_subsequence(codeword, N):
#     """
#     generate random seed, pick random number from 0 to 254
#     """
#     first_index = random.randint(0, N)
#     print("Start index:", first_index)
#     subsequence = [0]*N
#     subsequence = np.roll(codeword, -first_index)
#     return subsequence[:255]
