
from enum import Enum, auto
import numpy as np
from detectors import common as cm

from detectors import med, glrt, mmed, ed, agm, hr, vd1, grcr, gid, pride, lmpit, cfcpsc, wcfcpsc

from fusionCenter import fusion_center as FC

class Techniques(Enum):
    AGM = auto()
    ED = auto()
    GID = auto()
    GLRT = auto()
    GRCR = auto()
    HR = auto()
    LMPIT = auto()
    MED = auto()
    MMED = auto()
    PRIDe = auto()
    VD1 = auto()
    CFCPSC = auto()
    WCFCPSC = auto()


class Detector:
    def __init__(this, technique: Techniques, Sigma2=None, nSubBands=None):
        if ((technique == Techniques.ED or technique == Techniques.MED) and (np.all(Sigma2 == None))):
            raise ValueError(
                "This techniques (ED or MED) needs to inform 'Sigma2'")
        if ((technique == Techniques.CFCPSC or technique == Techniques.WCFCPSC) and (np.all(nSubBands == None))):
            raise ValueError(
                "This techniques (CFCPSC or WCFCPSC) needs to inform 'nSubBands'")
        this.technique = technique
        this.Sigma2 = Sigma2
        this.nSubBands = nSubBands

    # Param
    #   X   =   dataset, shape (n_rouns, m, l, n)
    # Return
    #   T   =   test statics, shape (n_rouns, m)
    def test_statistics_multi_antenna(this, X):
        N_elem, m, _, _ = X.shape
        T = np.zeros((N_elem, m))
        for i in range(0, m):
            T[:, i] = this.test_statistics(X[:, i, :, :])
        return T

    # Param
    #   X   =   dataset, shape (n_rouns, m ou l, n)
    # Return
    #   T   =   test statics, shape (n_rouns)
    def test_statistics(this, X):
        N_elem = X.shape[0]
        T = np.zeros((N_elem))
        for i in range(0, N_elem):
            T[i] = this.test_statistic(X[i, :, :])
        return T
    
    # Param
    #   X   =   dataset, shape (m ou l, n)
    # Return
    #   T   =   test static
    def test_statistic(this, X):
        if (this.technique == Techniques.AGM):
            return agm.test_statistic(X)
        elif (this.technique == Techniques.ED):
            return ed.test_statistic(X, this.Sigma2)
        elif (this.technique == Techniques.GID):
            return gid.test_statistic(X)
        elif (this.technique == Techniques.GLRT):
            return glrt.test_statistic(X)
        elif (this.technique == Techniques.GRCR):
            return grcr.test_statistic(X)
        elif (this.technique == Techniques.HR):
            return hr.test_statistic(X)
        elif (this.technique == Techniques.LMPIT):
            return lmpit.test_statistic(X)
        elif (this.technique == Techniques.MED):
            return med.test_statistic(X, np.mean(this.Sigma2))
        elif (this.technique == Techniques.MMED):
            return mmed.test_statistic(X)
        elif (this.technique == Techniques.PRIDe):
            return pride.test_statistic(X)
        elif (this.technique == Techniques.VD1):
            return vd1.test_statistic(X)
        elif (this.technique == Techniques.CFCPSC):
            return cfcpsc.test_statistic(X, this.nSubBands)
        elif (this.technique == Techniques.WCFCPSC):
            return wcfcpsc.test_statistic(X, this.nSubBands)

    def evaluate_dataset(this, X, Y, Npt=40):
        T = this.test_statistics(X)

        Pfa, Pd, AUC = cm.probabilities(T, Y, Npt)

        if(this.technique == Techniques.HR or this.technique == Techniques.VD1):
            return 1-Pfa, 1-Pd, 1-AUC

        return Pfa, Pd, AUC

    # Param
    #   X   =   dataset, shape (n_rouns, m, n)
    # Return
    #   T   =   test statics, shape (n_rouns, n_subbands)
    def test_statistics_with_subbands(this, X):
        if np.all(this.nSubBands == None):
            raise ValueError("This techniques (CFCPSC or WCFCPSC) needs to inform 'nSubBands' to calc test statistic")

        N_elem = X.shape[0]
        T = np.zeros((N_elem, this.nSubBands))
        for i in range(0, N_elem):
            T[i] = this.test_statistic(X[i, :, :])
        return T

    def evaluate_dataset_with_subbands(this, X, Y, Npt=40):
        T = this.test_statistics_with_subbands(X)

        subbands_decisions = this.decision_over_thresholds(T, Npt=Npt)

        SUs_decisions = FC.test_statistic_OR(subbands_decisions)
        Pfa, Pd, AUC = FC.roc_curve(SUs_decisions, Y)

        return Pfa, Pd, AUC


    # Take decision of each SU over different thresholds
    # Param
    #   SU_scores       = Test statistic in each SU, shape (n_rounds, m)
    #   Npt             = Number of threshold to check the decision
    # Return
    #   SUs_decisions   = Decision of each SU over Npt threshold, shape(n_rounds, m, Npt)
    def decision_over_thresholds(this, SU_statistics, Npt=40):
        N_elem, m = SU_statistics.shape

        Min = np.mean(SU_statistics[:, 0]) - 3 * np.std(SU_statistics[:, 0])
        Max = np.mean(SU_statistics[:, 0]) + 3 * np.std(SU_statistics[:, 0])
        SU_thresholds = np.linspace(Min, Max, Npt)
        SUs_decisions = np.zeros((N_elem, Npt, m))

        for run in range(0, N_elem):
            scores = SU_statistics[run, :]  # score of m SUs
            if(this.technique == Techniques.HR or this.technique == Techniques.VD1):
                # compare with thresholds to make SUs decisions
                SUs_decisions[run, :, :] = np.array(
                    [[1 if i < th else 0 for i in scores] for th in SU_thresholds])
            else:
                # compare with thresholds to make SUs decisions
                SUs_decisions[run, :, :] = np.array(
                    [[1 if i > th else 0 for i in scores] for th in SU_thresholds])

        SUs_decisions = np.swapaxes(SUs_decisions, 1, 2)
        return SUs_decisions
