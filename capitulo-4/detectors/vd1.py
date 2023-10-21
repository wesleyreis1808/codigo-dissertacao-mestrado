import numpy as np
from detectors import common as cm
from detectors import detector as det

def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.VD1)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x):
    m, _ = x.shape
    R = cm.scm(x)

    # VD1 (Volume-based detector) statistic:
    d = np.zeros((m))
    for j in range(0, m):
        d[j] = np.linalg.norm(R[j, :])
    DD = np.diag(d)
    T = np.real(np.log(np.linalg.det(np.linalg.inv(DD).dot(R))))
    return T
