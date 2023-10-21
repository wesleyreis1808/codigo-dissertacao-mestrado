import os, sys
from cmath import inf

# Set path to find custom modules
sys.path.append( os.path.dirname(os.path.dirname(  os.path.abspath(__file__) ) ) )

from dataset import dataset
####################################################################################################

relative_work_directory = "LOCAL_FILES/datasets/"
#################################################################################################### Create work directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
absolute_work_directory = os.path.join(parent_dir, relative_work_directory) 
if not os.path.exists(absolute_work_directory):
    os.makedirs(absolute_work_directory, exist_ok = True) 

####################################################################################################
scenaries = [
    {  # scenary 1 - selectivity, uniform dynamic noise and no shadowing
        "runs": 15000,
        "PUsignal": 1,
        "s": 1,
        "m": 6,
        "n": 160,
        "l": 1,
        "SNR": -10,
        "SPS": 4,
        "p": 4,
        "Sigma2avg": 1,
        "rhoN": 0.25,           # rhoN = 0 (constant noise) or rhoN > 0 (dynamic noise)
        "fading": "Rayleigh",
        "meanK": -inf,          
        "stdK": 1,
        "randK": False,
        "selective": True,
        "shadowing": False,
        "noise": "uniform",
    },
    {  # scenary 2 - selectivity, uniform dynamic noise and shadowing
        "runs": 15000,
        "PUsignal": 1,
        "s": 1,
        "m": 6,
        "n": 160,
        "l": 1,
        "SNR": -10,
        "SPS": 4,
        "p": 4,
        "Sigma2avg": 1,
        "rhoN": 0.25,           # rhoN = 0 (constant noise) or rhoN > 0 (dynamic noise)
        "fading": "Rayleigh",
        "meanK": -inf,
        "stdK": 1,
        "randK": False,
        "selective": True,
        "shadowing": True,
        "noise": "uniform",
    },
    {  # scenary 3 - selectivity, non-uniform dynamic noise and shadowing
        "runs": 15000,
        "PUsignal": 1,
        "s": 1,
        "m": 6,
        "n": 160,
        "l": 1,
        "SNR": -10,
        "SPS": 4,
        "p": 4,
        "Sigma2avg": 1,
        "rhoN": 0.25,             # rhoN = 0 (constant noise) or rhoN > 0 (dynamic noise)
        "fading": "Rayleigh",
        "meanK": -inf,
        "stdK": 1,
        "randK": False,
        "selective": True,
        "shadowing": True,
        "noise": "non-uniform",
    },
]

####################################################################################################
for d in scenaries:
    dat = dataset.DataSetSelective(n=d["n"], m=d["m"], SNR=d["SNR"], Sigma2avg=d["Sigma2avg"],
                                   rhoN=d["rhoN"], PUsignal=d["PUsignal"], SPS=d["SPS"], shadowing=d["shadowing"], p=d["p"], noiseType=d["noise"])

    X, Y, SNR = dat.generate_H0_H1(d["runs"])

    print("Dataset generated.\t\nSNR: %.3f" % SNR)
    dat.save(absolute_work_directory, X, Y)
