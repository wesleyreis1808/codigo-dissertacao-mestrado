import numpy as np


# Params
#   s               = Number of PU transmitters.
#   n               = Number of samples collected by SU
#   SNR             = signal to noise ratio in dB
#   Sigma2avg       = Average noise power
# Return
#   Complex iid Gaussian PU signal size(s,n)
def gaussian_signal(s, n, SNR,  Sigma2avg=1):
    Ptx = source_power(s, SNR, Sigma2avg)
    S = generateCplxElem(mu=0, std=1 / np.sqrt(2), size=(s, n))

    St = np.transpose(S)
    pwr = np.diag(np.concatenate((np.sqrt(Ptx))))
    S = np.transpose(np.matmul(St, pwr))
    return S


# Params
#   s               = Number of PU transmitters.
#   n               = Number of samples collected by SU
#   Ptx             = transmission power of each PU
#   T               = Number of samples per QPSK PU symbol (n/T must be integer).
#   Sigma2avg       = Average noise power
# Return
#   QPSK PU signal with T samples per symbol and size(s,n)
def qpsk_signal(s, n, SNR, T,  Sigma2avg=1):
    Ptx = source_power(s, SNR, Sigma2avg)
    S = []
    n_T = int(n / T)

    if n%T != 0:
        n_T = n_T + 1

    for symb in range(0, n_T):
        o = np.ones((1, int(T)))
        r = np.random.choice([-1, 1], size=(s, 1))
        i = np.random.choice([-1, 1], size=(s, 1))

        sample = np.matmul(r, o) + 1j * np.matmul(i, o)

        if(len(S) == 0):
            S = sample
        else:
            S = np.concatenate((S, sample), axis=1)

    S = S[:, 0:n]

    St = np.transpose(S)
    pwr = np.diag(np.concatenate((np.sqrt(Ptx / 2))))
    S = np.transpose(np.matmul(St, pwr))
    return S


# Params
#   m           = Number of SU receivers
#   SNR         = Signal to noise ratio in dB
#   Sigma2avg   = Average noise power
# Return
#    Transmission power of each PU size(s,1)
def source_power(s, SNR, Sigma2avg=1):
    # Average received power
    PRxavg = receive_power_avg(SNR, Sigma2avg)
    # Source powers
    Ptx = np.ones((s, 1)) * PRxavg / s
    return Ptx


# Params
#   m           = Number of SU receivers
#   SNR         = Signal to noise ratio in dB
#   rhoP        = Fraction of signal power variations about the mean
#   Sigma2avg   = Average noise power
# Return
#   Received powers size(m,1)
def receive_power(m, SNR, rhoP=0, Sigma2avg=1):
    # Average received power
    PRxavg = receive_power_avg(SNR, Sigma2avg)
    # Received powers (m x 1) variable over all sensing rounds
    PRx = np.random.rand(m, 1) * (2 * rhoP) + (1 - rhoP)
    # Normalize to mean = PRxavg
    PRx = PRx / np.mean(PRx) * PRxavg
    return PRx


def receive_power_avg(SNR, Sigma2avg=1):
    # Average received power
    PRxavg = Sigma2avg * (10 ** (SNR / 10))
    return PRxavg

def generateCplxElem(mu, std, size):
    return np.random.normal(mu, std, size) + 1j * np.random.normal(mu, std, size)
