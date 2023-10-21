import numpy as np

# Params
#   m           = Number of SU receivers
#   n           = Number of samples per SU
#   rhoN        = Fraction of noise power variations about the mean.
#   Sigma2avg   = Average noise power.
# Return
#   Gaussian noise matrices size (m,n)
def awgn_noise(m, n, Sigma2):
    W = np.zeros((m, n), dtype=complex)
    for j in range(0, m):
        W[j, :] = generate_complex_elem(Sigma2[j], size=(1, n))
    return W

# Params
#   m           = Number of SU receivers
#   rhoN        = Fraction of noise power variations about the mean.
#   Sigma2avg   = Average noise power.
# Return
#    Noise variances over each SU size(m,1)
def noise_variances(m, rhoN=0, Sigma2avg=1):
    if rhoN < 0  or rhoN > 1:
        raise ValueError("RhoN must be in [0,1]")
    
    Sigma2 = np.random.rand(m, 1) * (2 * rhoN) + (1 - rhoN)
    # Normalize to mean = Sigma2avg
    Sigma2 = (Sigma2 / np.mean(Sigma2)) * Sigma2avg
    return Sigma2

# Params
#   m           = Number of SU receivers
#   rhoN        = Fraction of noise power variations about the mean.
#   Sigma2avg   = Average noise power.
#   type        = Type of noise (uniform, non-uniform)
# Return
#    Noise variances over each SU size(m,1)
def noise_variances_type(m, rhoN=0, Sigma2avg=1, type="uniform"):
    # Noise uniform and constant
    if type == "uniform" and rhoN == 0:
        Sigma2 = np.ones((m,1)) * Sigma2avg
        return Sigma2
    
    # Noise uniform and dynamic
    elif type == "uniform" and rhoN > 0 and rhoN <= 1:
        Sigma2 = np.random.rand(m, 1) * (2 * rhoN) + (1 - rhoN)
        # Normalize to mean = Sigma2avg
        Sigma2 = (Sigma2 / np.mean(Sigma2)) * Sigma2avg
        return Sigma2
    
    # Noise non-uniform and constant
    elif type == "non-uniform" and rhoN == 0 and m <=6:
        Sigma2 = np.array([0.8, 0.9, 0.95, 1.1, 0.85, 1.15])
        Sigma2 = Sigma2[0:m]
        Sigma2 = (Sigma2 / np.mean(Sigma2)) * Sigma2avg
        return Sigma2.reshape(m,1)
    
    # Noise non-uniform and dynamic
    elif type == "non-uniform" and rhoN > 0 and rhoN <= 1 and m <=6:
        Sigma2 = np.array([0.8, 0.9, 0.95, 1.1, 0.85, 1.15])
        Sigma2 = Sigma2[0:m]
        Sigma2 = Sigma2 * (np.random.rand(m) * (2 * rhoN) + (1 - rhoN))
        Sigma2 = (Sigma2 / np.mean(Sigma2)) * Sigma2avg
        return Sigma2.reshape(m,1)

    else:
        raise ValueError("Type of noise must be 'uniform' or 'non-uniform' and rhoN must be in [0,1], and for 'non-uniform' m<=6")


def generate_complex_elem(sigma2, size):
    mean = 0
    std = np.sqrt(sigma2 / 2)
    return np.random.normal(mean, std, size) + 1j * np.random.normal(mean, std, size)
