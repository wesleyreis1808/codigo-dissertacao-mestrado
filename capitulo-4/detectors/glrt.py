import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.GLRT)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x):
    m, _ = x.shape
    R = cm.scm(x)
    lbda = cm.eig(R)
    # GLRT (generalized likelihood ratio test) statistic
    #T = lbda[0] / np.sum(lbda[np.arange(1, m)])
    T = lbda[0] / np.mean(lbda)
    return T
