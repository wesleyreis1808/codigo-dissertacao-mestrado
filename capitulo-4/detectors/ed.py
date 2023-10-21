import numpy as np
from detectors import detector as det

def evaluate_dataset(X, Y, Sigma2, Npt=40):
    d = det.Detector(det.Techniques.ED, Sigma2)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x, Sigma2):
    m, _ = x.shape
    # ED (energy detection) statistic
    SUM = 0
    for c in range(0, m):
        SUM = SUM + np.sum(np.abs(x[c, :]) ** 2) / (Sigma2[c])
    T = SUM
    return T
