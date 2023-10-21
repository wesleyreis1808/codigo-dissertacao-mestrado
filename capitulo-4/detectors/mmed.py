import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.MMED)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x):
    m, _ = x.shape
    R = cm.scm(x)
    lbda = cm.eig(R)
    # ER (eigenvalue ratio) statistic:
    T = lbda[0] / lbda[m-1]
    return T
