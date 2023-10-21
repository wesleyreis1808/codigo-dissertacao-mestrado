import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Sigma2avg=1, Npt=40):
    d = det.Detector(det.Techniques.MED, Sigma2avg)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x, Sigma2avg):
    R = cm.scm(x)
    lbda = cm.eig(R)
    # RLRT (Roy's largest root test) statistic
    T = lbda[0] / Sigma2avg
    return T
