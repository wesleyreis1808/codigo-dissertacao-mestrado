import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.AGM)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x):
    m, _ = x.shape
    R = cm.scm(x)
    lbda = cm.eig(R)
    # AGM (arithmetic to geometric np.mean detector) statistic
    T = np.sum(lbda) / ((np.prod(lbda)) ** (1 / m))
    return T
