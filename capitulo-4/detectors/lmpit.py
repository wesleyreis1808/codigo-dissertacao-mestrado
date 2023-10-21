import numpy as np
from detectors import common as cm
from detectors import detector as det


def evaluate_dataset(X, Y, Npt=40):
    d = det.Detector(det.Techniques.LMPIT)
    return d.evaluate_dataset(X, Y, Npt)


def test_statistic(x):
    R = cm.scm(x)
    # LMPIT (locally most powerful invariant test) statistic
    # Ref: The locally most powerful test for multiantenna spectrum sensing with uncalibrated receivers
    D = np.diag(np.diag(R))

    D_inv = np.linalg.inv(D) ** (1/2)
    C = D_inv.dot(R).dot(D_inv)

    T = np.linalg.norm(C, 'fro') ** 2
    return T