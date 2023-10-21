from math import floor
import numpy as np
import numpy.matlib
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, nSubBands=5, Npt=40):
    d = det.Detector(det.Techniques.WCFCPSC, nSubBands=nSubBands)
    return d.evaluate_dataset_with_subbands(X, Y, Npt)


def test_statistic(x, nSubBands):

    dim1Size, dim2Size = x.shape

    # CFCPSC number of samples per sub-band
    Vcfcpsc = floor(dim2Size/(2*nSubBands))

    # Step 1) PSDE
    Fline = abs(np.fft.fft(x, n=dim2Size, axis=1))**2/dim2Size
    #print("Fline ", Fline)
    # print()

    # Step 2) modified circular-even component of F'u
    F_aux = np.ones(shape=(dim1Size, dim2Size))
    F_aux[:, 0] = (Fline[:, 0] + Fline[:, int(dim2Size/2)])/2

    k = np.arange(1, dim2Size, 1)
    indices = (dim2Size-k)
    F_aux[:, 1:dim2Size] = (Fline[:, k] + Fline[:, indices])/2
    F = F_aux[:, np.arange(0, dim2Size/2, 1, dtype=int)]

    # print("F_u(k) ", F)
    # print("")

    # Step 3) Divide the sensed band into 'nSubBands' sub-bands and compute the signal power in the l-th sub-band, ell = 1, 2, ..., nSubBands, as
    Fell = np.zeros((dim1Size, nSubBands))
    for ell in range(0, nSubBands):
        for k in range(0, Vcfcpsc):
            Fell[:, ell] = Fell[:, ell] + F[:, ell*Vcfcpsc + k]

    # print("Fell ", Fell)
    # print("")

    # Step 4) Compute the total signal power in the sensed band,
    Ffull = np.zeros((dim1Size,1))
    Ffull[:,0] = np.sum(F[:, np.arange(0, dim2Size/2, 1, dtype=int)], axis=1)
    # print("Ffull ", Ffull)
    # print("")

    # Step 5) Compute the average of the ratio Fell_u/Ffull_u, where the noise variance influence is canceled-out, yielding
    r = Fell/(Ffull * np.ones((1, nSubBands)))

    # print("r ", r)
    # print("")

    # Step 6) For both the partial and the total sample fusion strategies, the adapted CF-CPSC test statistic for the ell-th subband is formed at the FC, yielding
    
    weights = np.matlib.repmat(((nSubBands - np.arange(1,nSubBands+1,1)) + 1) / nSubBands, dim1Size, 1)
    
    ravg = sum(r * weights)

    # print("ravg ", ravg)
    # print("")

    return ravg
