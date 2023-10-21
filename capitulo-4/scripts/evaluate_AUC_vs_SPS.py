# To find the dataset module
from cmath import inf
import sys, os, json

# Set path to find custom modules
sys.path.append( os.path.dirname(os.path.dirname(  os.path.abspath(__file__) ) ) )

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tensorflow.keras.models import load_model

from dataset import dataset
from dataset.noise import noise as noise
from detectors import  gid, pride, wcfcpsc, glrt
from detectors import common as cm

relative_work_directory = "LOCAL_FILES"
#################################################################################################### Create work directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
absolute_work_directory = os.path.join(parent_dir, relative_work_directory) 

dataset_directory = absolute_work_directory + "/datasets/"
model_directory = absolute_work_directory + "/models/"
result_directory = absolute_work_directory + "/result/evaluate_AUC_vs_SPS/"

if not os.path.exists(model_directory):
    os.makedirs(model_directory, exist_ok = True)
if not os.path.exists(result_directory):
    os.makedirs(result_directory, exist_ok = True)

#################################################################################################### Configurations
datasetsFile = [
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(2)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(3)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(4)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(5)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(6)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(7)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(8)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(9)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(10)_shadowing(True)_p(4)_noise(non-uniform)',
]

model_name = "DenseNet201_FFT"
####################################################################################################
######################################################################################### Load Model
try:
    modelpath = model_directory + model_name + ".h5"
    print("Loading model: ", modelpath)
    model = load_model(modelpath)
except:
    sys.exit("Model not found")

####################################################################################################

for i, scenario in enumerate(datasetsFile):
    dic = {}
    ################################################################################################ Pre-process Dataset
    X, Y = dataset.loadDataset(filename= dataset_directory + scenario +".mat")

    p1 = dataset.calcFFT(X)


    _pX, pY, input_shape = dataset.cplxToColum_and_hotEncoded(p1,Y)

    pX = dataset.reshapeDataset(_pX, 36,160)
    input_shape = (36,160,2)

    ################################################################################################ Evaluate with full dataset
    # _, accuracy = model.evaluate(pX, pY, verbose=0)
    # print("accuracy full dataset: ", accuracy)

    ################################################################################################ Test Model
    y_pred = model.predict(pX).ravel()
    aux_y = pY.ravel()

    Pfa, Pd, thresholds = roc_curve(aux_y, y_pred)
    AUC = auc(Pfa, Pd)
    Pfa, Pd = cm.subsamplingROC(Pfa, Pd, Npt=40)

    dic[model_name] = AUC
    ################################################################################################ Compare results
    Pfa, Pd, AUC = wcfcpsc.evaluate_dataset(X, Y, nSubBands=5, Npt=40)
    dic["WCFCPSC"] = AUC
   
    Pfa, Pd, AUC = gid.evaluate_dataset(X, Y, 40)
    dic["GID"] = AUC

    Pfa, Pd, AUC = pride.evaluate_dataset(X, Y, 40)
    dic["PRIDe"] = AUC
    
    Pfa, Pd, AUC = glrt.evaluate_dataset(X, Y, 40)
    dic["GLRT"] = AUC

    dic["SPS"] = int(scenario.split("SPS(")[1].split(")")[0])
    
    print("%d/%d Evaluate model '%s' over dataset '%s'\n" % (i+1, len(datasetsFile), model_name, scenario))

    with open(result_directory + scenario +".txt", "w") as fp:
        json.dump(dic, fp)  # encode dict into JSON
        print("Results saved in: ", result_directory )
        os._exit(0)


################################################################################################ 
################################################################################################ Plot Chart
import matplotlib.pyplot as plt
import numpy as np

datasetsFile = [
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(2)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(3)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(4)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(5)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(6)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(7)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(8)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(9)_shadowing(True)_p(4)_noise(non-uniform)',
    'selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(10)_shadowing(True)_p(4)_noise(non-uniform)',
]
ROCs = []
for i, scenario in enumerate(datasetsFile):
    with open(result_directory + scenario +".txt", "r") as fp:
        ROCs.append(json.load(fp))

print(ROCs)

with open(result_directory + "compiled_results.txt", "w") as fp:
    json.dump(ROCs, fp)  # encode dict into JSON
    print("Compiled results saved in: ", result_directory)

# SAMPLE
# dic = [ { 'SPS': 2,'DenseNet201_FFT_160': 0.9123510833333334,  'WCFCPSC': 0.7464161088888889, 'GID': 0.6407737177777779, 'PRIDe': 0.7250343777777777}, 
#         { 'SPS': 3,'DenseNet201_FFT_160': 0.9463365250000001, 'WCFCPSC': 0.8720523466666668, 'GID': 0.6509908866666666, 'PRIDe': 0.7450807022222222}, 
#       ]

import matplotlib.pyplot as plt
axi = [x["SPS"] for x in ROCs]
plt.figure(2)
plt.grid(True)
plt.plot(axi,[x["DenseNet201_FFT"] for x in ROCs], label="DenseNet201_FFT")
plt.plot(axi,[x["WCFCPSC"] for x in ROCs], label="WCFCPSC")
plt.plot(axi,[x["GID"] for x in ROCs], label="GID")
plt.plot(axi,[x["PRIDe"] for x in ROCs], label="PRIDe")
plt.plot(axi,[x["GLRT"] for x in ROCs], label="GLRT")
# plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(result_directory + "AUC_vs_SPS.png")
plt.show()
