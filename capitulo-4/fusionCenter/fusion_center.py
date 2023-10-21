import numpy as np

# Take FC decision
# Param
#   SUs_decisions   = Decision of each SUs, shape(n_rounds, m, Npt)
#   threshold       = threshold to take decision
# Return
#   FC_decisions    = The FC decision taked over SUs_decisions, shape(n_rounds, Npt)
def test_statistic(SUs_decisions, threshold):
    N_elem, _, Npt = SUs_decisions.shape

    FC_decisions = np.zeros((N_elem, Npt))
    for run in range(0, N_elem):
        SUs_decision = SUs_decisions[run, :, :].sum(axis=0)
        FC_decisions[run, :] = np.array([1 if i >= (threshold) else 0 for i in SUs_decision])# compare with threshold FC to make final decision

    return FC_decisions


# Calc Pd e Pfa
# Param
#   decisions       = Decisions taked
#   Y               = True decisions
# Return
#   Pfa             = Probability of false alarm
#   Pd              = Probability of detection
#   AUC             = Area under th ROC curve
def roc_curve(decisions, Y):
    N_elem, Npt = decisions.shape

    Pfa = np.zeros((Npt))
    Pd = np.zeros((Npt))

    N_h0 = len(Y[Y == 0])
    N_h1 = len(Y[Y == 1])

    for i in range(0, Npt):
        aux_h0 = 0
        aux_h1 = 0
        for ii in range(0, N_elem):
            if decisions[ii, i] == 1 and Y[ii] == 0:
                aux_h0 = aux_h0 + 1
            if decisions[ii, i] == 1 and Y[ii] == 1:
                aux_h1 = aux_h1 + 1
        Pfa[i] = aux_h0 / N_h0
        Pd[i] = aux_h1 / N_h1

    AUC = np.trapz(x=Pfa, y=Pd) * -1
    return Pfa, Pd, AUC


def test_statistic_AND(SUs_decisions):
    m = SUs_decisions.shape[1]
    return test_statistic(SUs_decisions, m)


def test_statistic_OR(SUs_decisions):
    return test_statistic(SUs_decisions, 1)


def test_statistic_MAJ(SUs_decisions):
    m = SUs_decisions.shape[1]
    return test_statistic(SUs_decisions, ((m+1)/2))
