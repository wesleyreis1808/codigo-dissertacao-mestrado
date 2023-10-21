# To find the dataset module
import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from dataset import dataset
from dataset.noise import noise as noise
from detectors import med, glrt, mmed, ed, agm, hr, vd1, grcr, gid, pride, lmpit, cfcpsc, wcfcpsc
from chart import chart


#################################################################################################### Load Dataset
datasetRelativePath = "LOCAL_FILES/datasets/selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(4)_shadowing(True)_p(4)_noise(non-uniform).mat"
parent_dir = os.path.dirname(path.dirname(path.abspath(__file__)))
datasetAbsolutePath = os.path.join(parent_dir, datasetRelativePath) 
X, Y = dataset.loadDataset(filename=datasetAbsolutePath)

#################################################################################################### Compare results
Sigma2 = noise.noise_variances_type(6, 0.8, 1, "uniform")

c = chart.Plot()

Pfa, Pd, AUC = ed.evaluate_dataset(X, Y, Sigma2, 40)
c.add_plot(Pfa, Pd, AUC, label="ED", style='m-*')
Pfa, Pd, AUC = med.evaluate_dataset(X, Y, 1, 40)
c.add_plot(Pfa, Pd, AUC, label="MED", style='r-^')
Pfa, Pd, AUC = mmed.evaluate_dataset(X,Y,40)
c.add_plot(Pfa, Pd, AUC, label="MMED", style='g-d')
Pfa, Pd, AUC = glrt.evaluate_dataset(X,Y,40)
c.add_plot(Pfa, Pd, AUC, label="GLRT", style='k-s')
Pfa, Pd, AUC = agm.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="AGM", style='m-p')
Pfa, Pd, AUC = hr.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="HR", style='b-s')
Pfa, Pd, AUC = vd1.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="VD1", style='b-v')
Pfa, Pd, AUC = grcr.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="GRCR", style='r-o')
Pfa, Pd, AUC = gid.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="GID", style='k-o')
Pfa, Pd, AUC = pride.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="PRIDe", style='m-o')
Pfa, Pd, AUC = lmpit.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="LMPIT", style='y->')
Pfa, Pd, AUC = cfcpsc.evaluate_dataset(X, Y, nSubBands=5, Npt=40)
c.add_plot(Pfa, Pd, AUC, label="CFCPSC", style='g-d')
Pfa, Pd, AUC = wcfcpsc.evaluate_dataset(X, Y, nSubBands=5, Npt=40)
c.add_plot(Pfa, Pd, AUC, label="WCFCPSC", style='b-v')
wcfcpsc_Pfa, wcfcpsc_Pd = Pfa, Pd

c.show()