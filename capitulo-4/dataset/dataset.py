import numpy as np
import scipy.io as scipy
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import cv2
from cmath import inf

from dataset.noise import noise as noise
from dataset.channel import channel as ch
from dataset.signal import signal as sig


class DataSet:
    def __init__(this, n: int = 60, m: int = 6, SNR: int = -10, s: int = 1, Sigma2avg: int = 1, rhoP: float = 0, rhoN: float = 0, meanK: float = -inf, stdK: float = 4.13, randK: int = 0, PUsignal: int = 0):
        this.s = s
        this.m = m
        this.SNR = SNR
        this.n = n
        this.T = this.n / 10
        this.Sigma2avg = Sigma2avg
        this.rhoP = rhoP
        this.rhoN = rhoN
        this.meanK = meanK
        this.stdK = stdK
        this.randK = randK
        this.PUsignal = PUsignal

    def generate_random(this, N_elem, p=[0.5, 0.5]):    
        X = np.zeros((N_elem, this.m, this.n), dtype=complex)
        Y = np.zeros((N_elem), dtype=int)
        PRx_measured = np.zeros((N_elem))
        Pnoise_measured = np.zeros((N_elem))

        # Average received power
        PRxavg = sig.receive_power_avg(this.SNR, this.Sigma2avg)

        for i in range(0, N_elem):
            # PU signal (sxn):
            if this.PUsignal == 0:
                S = sig.gaussian_signal(
                    this.s, this.n, this.SNR, this.Sigma2avg)
            else:
                S = sig.qpsk_signal(this.s, this.n, this.SNR,
                                    this.T, this.Sigma2avg)

            # Gaussian noise matrices (mxn):
            Sigma2 = noise.noise_variances(this.m, this.rhoN, this.Sigma2avg)
            W = noise.awgn_noise(this.m, this.n, Sigma2)

            PRx = sig.receive_power(
                this.m, this.SNR, this.rhoP, this.Sigma2avg)

            G = ch.channel_gain(PRx, PRxavg)
            if(this.meanK == -inf):
                A = ch.channel_rayleigh_fading(this.m, this.s)
            else:
                A = ch.channel_rice_fading(this.m, this.s, this.meanK, this.stdK, this.randK)
            H = np.matmul(G, A)

            # Received signal matrix
            rd = np.random.choice([0, 1], p=p)
            if (rd == 0):
                X[i, :, :] = W
                Y[i] = 0
            else:
                X[i, :, :] = np.matmul(H, S) + W
                Y[i] = 1

            # Signal and noise power measurements in each run
            PRx_measured[i] = (
                sum(sum(np.abs(np.matmul(H, S)) ** 2)) / (this.m * this.n))
            Pnoise_measured[i] = (sum(sum(np.abs(W) ** 2)) / (this.m * this.n))

        SNR = 10 * np.log10(np.mean(PRx_measured / Pnoise_measured))
        return X, Y, SNR

    def generate_H0_H1(this, N_elem):
        X = np.zeros((N_elem*2, this.m, this.n), dtype=complex)
        Y = np.zeros((N_elem*2), dtype=int)
        PRx_measured = np.zeros((N_elem))
        Pnoise_measured = np.zeros((N_elem))

        # Average received power
        PRxavg = sig.receive_power_avg(this.SNR, this.Sigma2avg)

        for index in range(0, N_elem):
            # PU signal (sxn):
            if this.PUsignal == 0:
                S = sig.gaussian_signal(
                    this.s, this.n, this.SNR, this.Sigma2avg)
            else:
                S = sig.qpsk_signal(this.s, this.n, this.SNR,
                                    this.T, this.Sigma2avg)

            # Gaussian noise matrices (mxn):
            Sigma2 = noise.noise_variances(this.m, this.rhoN, this.Sigma2avg)
            W0 = noise.awgn_noise(this.m, this.n, Sigma2)
            W1 = noise.awgn_noise(this.m, this.n, Sigma2)

            PRx = sig.receive_power(
                this.m, this.SNR, this.rhoP, this.Sigma2avg)

            G = ch.channel_gain(PRx, PRxavg)
            if(this.meanK == -inf):
                A = ch.channel_rayleigh_fading(this.m, this.s)
            else:
                A = ch.channel_rice_fading(this.m, this.s, this.meanK, this.stdK, this.randK)
            H = np.matmul(G, A)

            # Received signal matrix
            i = index*2
            # Over H0
            X[i, :, :] = W0
            Y[i] = 0
            # Over H1
            X[i+1, :, :] = np.matmul(H, S) + W1
            Y[i+1] = 1

            # Signal and noise power measurements in each run
            PRx_measured[index] = (
                sum(sum(np.abs(np.matmul(H, S)) ** 2)) / (this.m * this.n))
            Pnoise_measured[index] = (
                sum(sum(np.abs(W0) ** 2)) / (this.m * this.n))

        SNR = 10 * np.log10(np.mean(PRx_measured / Pnoise_measured))
        return X, Y, SNR

    def save(this, path, X, Y):
        mdic = {"dataX": X, "dataY": Y}
        filename = this.__get_filename()
        path_file = path + filename + ".mat"
        scipy.savemat(path_file, mdic)
        return path_file

    def __get_filename(this):
        filename = "n(%d)_m(%d)_SNR(%d)_s(%d)_Sigma2avg(%d)_rhoP(%.2f)_rhoN(%.2f)_meanK(%.2f)_stdK(%.2f)_randK(%d)_PUsignal(%d)" % (
            this.n, this.m, this.SNR, this.s, this.Sigma2avg, this.rhoP, this.rhoN, this.meanK, this.stdK, this.randK, this.PUsignal)
        return filename

class DataSetSelective:
    def __init__(this, n: int = 60, m: int = 6, SNR: int = -10, Sigma2avg: int = 1, rhoN: float = 0, PUsignal: int = 0, SPS: int = 5, shadowing: bool = False, p: int = 4, noiseType: str = "uniform"):
        this.s = 1
        this.m = m
        this.SNR = SNR
        this.n = n
        this.T = SPS
        this.Sigma2avg = Sigma2avg
        # this.rhoP = rhoP
        this.rhoN = rhoN
        this.PUsignal = PUsignal
        this.shadowing = shadowing
        this.p = p                  # Number of Z paths derivations
        this.noiseType = noiseType  # Noise type: uniform or non-uniform
    
    def generate_H0_H1(this, N_elem):
        X = np.zeros((N_elem*2, this.m, this.n), dtype=complex)
        Y = np.zeros((N_elem*2), dtype=int)
        PRx_measured = np.zeros((N_elem))
        Pnoise_measured = np.zeros((N_elem))

        # Average received power
        # PRxavg = sig.receive_power_avg(this.SNR, this.Sigma2avg)

        for index in range(0, N_elem):
            # PU signal (sxn):
            if this.PUsignal == 0:
                S = sig.gaussian_signal(this.s, this.n, this.SNR, this.Sigma2avg)
            else:
                S = sig.qpsk_signal(this.s, this.n, this.SNR, this.T, this.Sigma2avg)

            # Gaussian noise matrices (mxn):
            Sigma2 = noise.noise_variances_type(this.m, this.rhoN, this.Sigma2avg, this.noiseType)
            W0 = noise.awgn_noise(this.m, this.n, Sigma2)
            W1 = noise.awgn_noise(this.m, this.n, Sigma2)

            # PRx = sig.receive_power(this.m, this.SNR, this.rhoP, this.Sigma2avg)

            # Selective chanel (m x p)
            H = ch.channel_rayleigh_selective_shadowing(m=this.m, s=this.s, p=this.p, isShadowing=this.shadowing)

            # Convolute signal
            HS = ch.convolute_signal_with_selective_channel(S, H)

            # Received signal matrices under H0 and H1 (mxn):
            X_h0 = W0
            X_h1 = HS + W1

            # Received signal matrix
            i = index*2
            # Over H0
            X[i, :, :] = X_h0
            Y[i] = 0
            # Over H1
            X[i+1, :, :] = X_h1
            Y[i+1] = 1

            # Signal and noise power measurements in each run
            PRx_measured[index] = (sum(sum(np.abs(HS) ** 2)) / (this.m * this.n))
            Pnoise_measured[index] = (sum(sum(np.abs(W0) ** 2)) / (this.m * this.n))

        SNR = 10 * np.log10(np.mean(PRx_measured / Pnoise_measured))
        return X, Y, SNR
    
    def save(this, path, X, Y):
        mdic = {"dataX": X, "dataY": Y}
        filename = this.__get_filename()
        path_file = path + filename + ".mat"
        scipy.savemat(path_file, mdic)
        return path_file

    def __get_filename(this):
        filename = "selective_n(%d)_m(%d)_SNR(%d)_Sigma2avg(%d)_rhoN(%.2f)_PUsignal(%d)_SPS(%d)_shadowing(%r)_p(%d)_noise(%s)" % (
            this.n, this.m, this.SNR, this.Sigma2avg, this.rhoN, this.PUsignal, this.T, this.shadowing, this.p, this.noiseType)
        return filename

def loadDataset(filename):
    # Carregar dataset
    mat = scipy.loadmat(filename)
    X = mat['dataX']
    Y = mat['dataY']

    # Transform Y in a vector
    Y = np.concatenate(Y)

    X, Y = shuffle(X, Y)
    return X, Y


def cplxToColum_and_hotEncoded(X, Y):
    dim = (X.shape[1], X.shape[2], 2)
    return convertComplexToColumns(X), convertLabelToHotEncoded(Y), dim


def calcSCM(X):
    num_elem, m, n = X.shape
    nX = np.zeros((num_elem, m, m), dtype=complex)
    for i in range(0, num_elem):
        aux = X[i, :, :]
        #nX[i,:,:] = cm.scm(aux)
        nX[i, :, :] = np.cov(X[i, :, :], rowvar=True, bias=True)

    return nX

def calcFFT(X):
    num_elem, m, n = X.shape
    nX = np.zeros((num_elem, m, n), dtype=complex)
    for i in range(0, num_elem):
        aux = X[i, :, :]
        nX[i, :, :] = np.fft.fft(aux, n=n, axis=1)

    return nX

def convertComplexToColumns(X):
    # E = numero de eventos
    # M = numero de CRs
    # N = numero de amostras

    # Muda tamanho da matris de ExMxN para ExMxNx1
    aux_X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    # Decompoe os numeros complexos em sua parte real e imaginaria
    columns = (np.real(aux_X), np.imag(aux_X))
    # Concatena a parte real e imaginaria a matris ficando assim ExMxNx2
    _X = np.concatenate(columns, axis=3)
    return _X

def convertLabelToHotEncoded(Y):
    # E = numero de eventos

    # Transpoe a matrix Y de 1xE para Ex1
    # _Y = np.transpose(Y)
    # # Remove a segunda dimensao, transformando a matris em um vetor
    # _Y = _Y[:, 0]
    # Modifica a saida binaria para a representacao hot_enconded
    Y_hot_encoded = to_categorical(Y)

    return Y_hot_encoded

def splitDataset(X, Y, train_size=0.6, val_size=0.2, show=True):
    total_num = Y.shape[0]

    train_num = int(total_num * train_size)
    val_num = int(total_num * val_size)

    x_train = X[0:train_num]
    y_train = Y[0:train_num]

    x_val = X[train_num: train_num+val_num]
    y_val = Y[train_num: train_num+val_num]

    x_test = X[train_num+val_num:]
    y_test = Y[train_num+val_num:]

    if(show):
        print("Training data:", x_train.shape)
        print("Training labels:", y_train.shape)
        print("Validation data:", x_val.shape)
        print("Validation labels:", y_val.shape)
        print("Testing data", x_test.shape)
        print("Testing labels", y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test

def reshapeDatasetElem(elem, width: int, height: int):
    return cv2.resize(elem, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

def reshapeDataset(SCMs, width: int, height: int):
    num_elem, _, _, _ = SCMs.shape
    nSCM = np.zeros((num_elem, width, height, 2), dtype=float)

    for i in range(0, num_elem):
        nSCM[i, :, :, :] = reshapeDatasetElem(SCMs[i, :, :, :], width, height)

    return nSCM

def add3Channel(X):
    num_elem, w, h, _ = X.shape
    nSCM = np.zeros((num_elem, w, h, 3), dtype=float)
    nSCM[:, :, :, 0] = X[:, :, :, 0]
    nSCM[:, :, :, 1] = X[:, :, :, 1]

    return nSCM
