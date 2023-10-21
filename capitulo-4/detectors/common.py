import math
import numpy as np

def scm(X):
    _, n = X.shape
    R = X.dot(conjugate_transpose(X)) / n
    return R


def conjugate_transpose(M):
    return np.transpose(np.conjugate(M))


def eig(M):
    v, _ = np.linalg.eig(M)
    r = np.real(v)
    r.sort()
    return np.flip(r)


def probabilities(T, Y, Npt=40):
    k = -1
    T_h0 = T[Y == 0]
    T_h1 = T[Y == 1]
    N_elem = int(Y.size/2)

    CDF_H0 = np.zeros((Npt))
    CDF_H1 = np.zeros((Npt))
    Min = np.mean(T_h0) - 3 * np.std(T_h0)
    Max = np.mean(T_h1) + 3 * np.std(T_h1)
    thresholds = np.linspace(Min, Max, Npt)

    for i in np.arange(0, Npt):
        aux_h0 = 0
        aux_h1 = 0
        k += 1
        for ii in range(0, N_elem):
            if T_h0[ii] < thresholds[i]:
                aux_h0 = aux_h0 + 1
            if T_h1[ii] < thresholds[i]:
                aux_h1 = aux_h1 + 1
        CDF_H0[k] = aux_h0 / N_elem
        CDF_H1[k] = aux_h1 / N_elem

    Pfa = 1 - CDF_H0
    Pd = 1 - CDF_H1

    AUC = np.trapz(x=Pfa, y=Pd) * -1
    return Pfa, Pd, AUC


def _probabilities(T, Y, Npt=40):
    min = np.min(T)
    max = np.max(T)
    thresholds = np.linspace(min, max, Npt)

    Pd = np.zeros((Npt))
    Pfa = np.zeros((Npt))

    i = 0
    for threshold in thresholds:
        j = 0
        for t in T:
            if(t > threshold and Y[j] == 0):  # false alarm
                Pfa[i] += 1
            elif(t > threshold and Y[j] == 1):  # detection
                Pd[i] += 1
            j += 1
        i += 1

    Pfa = Pfa/(Y[Y == 0].size)
    Pd = Pd/(Y[Y == 1].size)

    AUC = np.trapz(x=Pfa, y=Pd) * -1
    return Pfa, Pd, AUC

def subsamplingROC(Pfa, Pd, Npt=40):
    points = np.concatenate((Pfa.reshape(-1,1), Pd.reshape(-1,1)), axis=1)
    rows, _ = points.shape
    positions = np.zeros((Npt), dtype=int)
    curveSize = 0

    for i in range(1, rows):
        curveSize += math.dist(points[i-1], points[i])
    pointDistance = curveSize/(Npt-1)

    pin = 0
    positions[pin] = 0
    count = 1
    for i in range(1, rows):
        dist = math.dist(points[pin], points[i])
        if dist > pointDistance:
            positions[count] = i-1
            pin = i-1
            count += 1
            if count == Npt-1:
                break
    
    # print("count: ", count)
    for i in range(count, Npt):
        positions[i] = rows-1
    # print("positions: ", positions)
    # print("P0: ", points[positions[0]])  

    return Pfa[positions], Pd[positions]

