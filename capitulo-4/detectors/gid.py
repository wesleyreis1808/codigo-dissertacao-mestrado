import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.GID)
    return d.evaluate_dataset(X, Y, Npt)



def test_statistic(x):
    m, _ = x.shape
    R = cm.scm(x)
    # GID (Gini index detector) statistic
    Num = 0
    for u in range(0, m ** 2):
        for j in range(u, m ** 2):
            Num += np.abs(R.item(u) - R.item(j))
    T = (1 * (m ** 2 - m)) * np.sum(np.abs(R)) / Num
    return T
