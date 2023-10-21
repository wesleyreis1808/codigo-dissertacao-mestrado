import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.HR)
    return d.evaluate_dataset(X, Y, Npt)



def test_statistic(x):
    R = cm.scm(x)
    # HR (Hadamard ratio) statistic:
    T = np.real(np.linalg.det(R) / np.prod(np.diag(R)))
    return T
