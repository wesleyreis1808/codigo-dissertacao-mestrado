import numpy as np
from detectors import common as cm
from detectors import detector as det

def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.PRIDe)
    return d.evaluate_dataset(X, Y, Npt)

def test_statistic(x):
    R=cm.scm(x)
    # PRIDe (Pietra-Ricci index detector) statistic
    # https://en.wikipedia.org/wiki/Hoover_index
    T=np.sum(np.abs(R)) / np.sum(np.abs(R - np.mean(R)))
    return T
