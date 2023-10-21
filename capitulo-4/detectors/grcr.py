import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.GRCR)
    return d.evaluate_dataset(X, Y, Npt)



def test_statistic(x):
    R = cm.scm(x)
    # GRCR (Gershgorin radii and centers ratio) detector statistic
    T = np.sum((np.sum(np.abs(R), 2-1) - np.diag(R))) / np.sum(np.diag(R))
    return T