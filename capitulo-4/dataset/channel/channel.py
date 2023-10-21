import numpy as np
import numpy.matlib

# Return
#   Multipath Rice fading matrix. size(m,s)
def channel_rice_fading(m, s, meanK, stdK, randK=1):
    A = np.zeros((m, s), dtype=complex)
    for row in range(0, m):
        for col in range(0, s):
            if randK == 1:
                K = 10 ** (np.random.randn(1, 1) * stdK + meanK) / 10
            else:
                K = 10 ** (meanK / 10)
            A[row, col] = generateRiceFadingElem(K)
    return A

# Return
#   Multipath Rayleigh fading matrix. size(m,s)
def channel_rayleigh_fading(m, s):
    A = np.zeros((m, s), dtype=complex)
    for row in range(0, m):
        for col in range(0, s):
            A[row, col] = generateRayleighFadingElem()
    return A


# Generate Rayleigh fading element
# Return
#   Complex Rayleigh fading element
def generateRayleighFadingElem():
    return np.sqrt(1/2) * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))

# Generate Rice fading element
# Params
#   K       = Rice factor
# Return
#   Complex Rice fading element
def generateRiceFadingElem(K):
    mean = np.sqrt(K / (2 * (K + 1)))
    std = np.sqrt((1 - K / (K + 1)) / 2)

    return np.random.normal(mean, std) + 1j * np.random.normal(mean, std)

# Params
#   PRx     = Received powers (m x 1) variable over all sensing rounds
#   PRxavg  = Average received power
# Return
#   Matrix with channels gains size(m,m)
def channel_gain(PRx, PRxavg):
    # Channel matrix (mxs):
    G = np.sqrt((PRx / PRxavg))
    G = np.diag(np.concatenate(G))
    return G

# Generate channel with selectivity, Rayleigh fading and lognormal shadowing
# Params
#   m        = Number of SUs
#   s        = NUmber of PUs
#   p        = Number of paths (derivations)
#   isShadowing    = Enable or disable shadowing
#   sigma_dB = Standard deviation of the shadowing
# Return
#   Selective channel, with/without shadowing and Rayleigh fading with dimension complex(m x p)
def channel_rayleigh_selective_shadowing(m, s, p, isShadowing=False, sigma_dB=4):

    spaceDimension=60
    decorrelationDist=30

    if s > 1:
        raise Exception("Channel not implement for number os PU > 1 (s>1)")

    if isShadowing: #0 or -Inf (-Inf to cancel the shadowing)
        mu_dB = 0
    else:
        mu_dB = -np.Inf

    # HR - Rayleigh fading channel
    _, ngns = impulse_gains(p)
    HR = np.sqrt(0.5) * ( (np.random.randn(m, p) +  1j *np.random.randn(m, p)) * np.matlib.repmat(ngns, m, 1))

    # HLN - Log Normal shadowing channel
    corr = channel_shadowing_correlation(m, sigma_dB, mu_dB, L=spaceDimension, Dcorr=decorrelationDist)
    rep_coor = np.matlib.repmat(corr, 1, p)
    ln_elems = np.random.uniform(low=0.0, high=2*np.pi, size=(m, p))
    HLN = rep_coor * np.exp(-1j * ln_elems)
    
    H = HLN + HR
    mu = (mu_dB / (20 * np.log10(np.exp(1))))
    d0 = (sigma_dB / (20 * np.log10(np.exp(1))))**2
    G = 1 + p * np.exp(2 * mu + 2 * d0)
    H = H / np.sqrt(G)

    return H

# Gains of each path
#   p       = Number of paths
# Return
#   gns     = gains of exponencial impulse response, size(p)
#   ngns    = normalized gains, size(p)
def impulse_gains(p):
    gns = np.zeros((p))
    gns2 = np.zeros((p))
    for i in range(p):
        gns[i] = np.power(10, -(-20 * np.log10(np.exp(-i/1.303))/20))
        gns2[i] = gns[i]**2

    ngns = gns / np.sqrt(np.sum(gns2))
    return gns, ngns

# Generate lognormal shadowing
# Params
#   m        = Number of SUs
#   sigma_dB = Standard deviation of the shadowing
#   mu_dB    = 
#   L        = 3D area of SUs in meters
#   Dcorr    = Decorrelation distance
#   isCorr   = Generate shadowing with correlation?
#   N        = Number os samples of each SU
# Return
#   Lognormal shadowing with/witout correlation, with dimension complex(m x N)
def channel_shadowing_correlation(m, sigma_dB, mu_dB, L=60, Dcorr=30, isCorr=1, N=1):
    if isCorr == 0:
        return np.random.randn(m, N)*sigma_dB+mu_dB
    elif L/Dcorr <= 0.01:
        return np.matlib.repmat(np.random.rand(1, N)*sigma_dB + mu_dB, m, 1)
    else:
        # gera pontos descorrelacionados no grid nxn
        n = int(np.ceil(L/Dcorr) + 1)

        # gera pontos uniformemente distribuidos no espaco LxLxL
        xi = np.random.uniform(0, L, (1, m))
        yi = np.random.uniform(0, L, (1, m))
        zi = np.random.uniform(0, L, (1, m))

        HLN = np.zeros((m, N))

        z = np.random.randn(n, n, n, N) * sigma_dB
        # z = np.ones((n, n, n, N)) * sigma_dB

        for i in range(0, m):
            # cordenadas normalizadas
            X = xi.flatten()[i]/Dcorr
            Y = yi.flatten()[i]/Dcorr
            Z = zi.flatten()[i]/Dcorr

            # ponto descorrelacionado de referencia para x, y e z
            gridx = int(np.floor(X))
            gridy = int(np.floor(Y))
            gridz = int(np.floor(Z))

            # correcao das cordenadas
            X = X - np.floor(X)
            Y = Y - np.floor(Y)
            Z = Z - np.floor(Z)

            # erro na estremidade do grid, correcao necessaria
            if gridx + 1 > n:
                gridx = n - 1
                X = X + 1

            # erro na estremidade do grid, correcao necessaria
            if gridy + 1 > n:
                gridy = n - 1
                Y = Y + 1

            # erro na estremidade do grid, correcao necessaria
            if gridz + 1 > n:
                gridz = n - 1
                Z = Z + 1
            
            # pega correspondentes pontos descorrelacionados
            Sa = np.reshape(a=z[gridx, gridy, gridz,:], newshape=(1, N))
            Sb = np.reshape(a=z[gridx + 1,gridy ,gridz,:], newshape=(1, N))
            Sc = np.reshape(a=z[gridx, gridy + 1, gridz,:], newshape=(1, N))
            Sd = np.reshape(a=z[gridx + 1,gridy + 1, gridz,:], newshape=(1, N))
            Se = np.reshape(a=z[gridx, gridy, gridz + 1,:], newshape=(1, N))
            Sf = np.reshape(a=z[gridx + 1,gridy, gridz + 1,:], newshape=(1, N))
            Sg = np.reshape(a=z[gridx, gridy + 1, gridz + 1,:], newshape=(1, N))
            Sh = np.reshape(a=z[gridx + 1, gridy + 1,gridz + 1,:], newshape=(1, N))

            # fator de normalizacao para que HG tenha sigma^2 = var
            G = np.sqrt((1 - 2*X + 2*(X**2)) * (1 - 2*Y + 2*(Y**2)) * (1 - 2*Z + 2*(Z**2)))
            HLN[i,:] = (((Sa*(1 - X) + Sb*X)*(1 - Y) + (Sc*(1 - X) + Sd*X)*Y)*(1 - Z) + ((Se*(1 - X) + Sf*X)*(1 - Y) + (Sg*(1 - X) + Sh*X)*Y)*Z)/G

        HLN = HLN + mu_dB

        HLN = 10**(HLN/20); # transforma de gaussiana para lognormal
        LN_phase = np.random.uniform(-np.pi, np.pi, (m, N)) # gera fase uniforme para HLN

        HLN = HLN * (np.cos(LN_phase) + 1j*np.sin(LN_phase))

        return HLN

# Signal over seletive channel
#   S       = Signal            (s x n)
#   H       = Selective channel (m x p)
# Return
#    (m x n)
def convolute_signal_with_selective_channel(S, H):
    _, n = S.shape
    m, _ = H.shape

    HS = np.zeros((m,n), dtype=complex)
    for i in range(0,m):
        HS[i, :] = np.convolve(S[0,:], H[i,:])[0:n]
    
    return HS