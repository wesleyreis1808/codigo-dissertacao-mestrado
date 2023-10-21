import os, sys

# Set path to find custom modules
sys.path.append( os.path.dirname(os.path.dirname(  os.path.abspath(__file__) ) ) )

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from dataset import dataset
from dataset.noise import noise as noise
from detectors import gid, pride, wcfcpsc, glrt
from chart import chart

which_scenary = 3 # 1: cenario1, 2: cenario2, 3: cenario3

relative_work_directory = "LOCAL_FILES"
#################################################################################################### Create work directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
absolute_work_directory = os.path.join(parent_dir, relative_work_directory) 

dataset_directory = absolute_work_directory + "/datasets/"
model_directory = absolute_work_directory + "/models/"
result_directory = absolute_work_directory + "/result/evaluate_detectors_over_noise_dynamic/"

if not os.path.exists(model_directory):
    os.makedirs(model_directory, exist_ok = True)
if not os.path.exists(result_directory):
    os.makedirs(result_directory, exist_ok = True)

#################################################################################################### Load Dataset
if which_scenary == 1:
    scenary = "cenario1"
    X, Y = dataset.loadDataset(filename= dataset_directory + "selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.25)_PUsignal(1)_SPS(4)_shadowing(False)_p(4)_noise(uniform).mat")
elif which_scenary == 2:
    scenary = "cenario2"
    X, Y = dataset.loadDataset(filename= dataset_directory + "selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.25)_PUsignal(1)_SPS(4)_shadowing(True)_p(4)_noise(uniform).mat")
elif which_scenary == 3:
    scenary = "cenario3"
    X, Y = dataset.loadDataset(filename= dataset_directory + "selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.25)_PUsignal(1)_SPS(4)_shadowing(True)_p(4)_noise(non-uniform).mat")

#################################################################################################### Pre-process Dataset
p1 = dataset.calcFFT(X)

_pX, pY, input_shape = dataset.cplxToColum_and_hotEncoded(p1,Y)
pX = dataset.reshapeDataset(_pX, 36,160)
input_shape = (36,160,2)
x_train, y_train, x_val, y_val, x_test, y_test = dataset.splitDataset(pX, pY)

#################################################################################################### Train Model
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from keras.applications.densenet import DenseNet201
####################################################################################################
keras_model = DenseNet201 
model_name = "DenseNet201_FFT"
retrain_model = False
reset_model = False

def get_compiled_model():
    base_model = keras_model(input_shape=input_shape, weights=None, include_top=False, classes=2)
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(50, activation='relu')
    dense_layer_2 = layers.Dense(20, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    # opt = Adam(lr=0.0001)
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model
#################################################################################################### Load Model
try:
    model = load_model(model_directory + model_name + ".h5")
    if reset_model:
        raise Exception("Retrain")
except:
    retrain_model = True
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = get_compiled_model()

#################################################################################################### Train Model
if retrain_model or reset_model :
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_directory + model_name  + ".h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    hist = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), batch_size=32, callbacks=[checkpoint, es])
    #chart.plotHistory(hist)

    # Load best model
    model = load_model(model_directory + model_name + ".h5")

_, accuracy = model.evaluate(pX, pY, verbose=0)
print("accuracy full dataset: ", accuracy)

#################################################################################################### Test Model
from detectors import common as cm

y_pred = model.predict(x_test).ravel() 
aux_y_test = y_test.ravel()

Pfa, Pd, thresholds = roc_curve(aux_y_test, y_pred)
AUC = auc(Pfa, Pd)
Pfa, Pd = cm.subsamplingROC(Pfa, Pd, Npt=40)
c = chart.Plot()

c.add_plot(Pfa, Pd, AUC, label=model_name, style='k-^')
rna_Pfa, rna_Pd, rna_AUC = Pfa, Pd, AUC

#################################################################################################### Compare results
Pfa, Pd, AUC = gid.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="GID", style='k-o')
gid_Pfa, gid_Pd, gid_AUC = Pfa, Pd, AUC

Pfa, Pd, AUC = pride.evaluate_dataset(X, Y, 40)
c.add_plot(Pfa, Pd, AUC, label="PRIDe", style='m-o')
pride_Pfa, pride_Pd, pride_AUC = Pfa, Pd, AUC

Pfa, Pd, AUC = wcfcpsc.evaluate_dataset(X, Y, nSubBands=5, Npt=40)
c.add_plot(Pfa, Pd, AUC, label="WCFCPSC", style='b-v')
wcfcpsc_Pfa, wcfcpsc_Pd, wcfcpsc_AUC = Pfa, Pd, AUC

Pfa, Pd, AUC = glrt.evaluate_dataset(X, Y, Npt=40)
c.add_plot(Pfa, Pd, AUC, label="GLRT", style='r-v')
glrt_Pfa, glrt_Pd, glrt_AUC = Pfa, Pd, AUC

c.save(result_directory + model_name + scenary)
# c.show()

## Save pd e pfa 
import scipy.io as scipy
mdic = {"rna_Pd": rna_Pd, "rna_Pfa": rna_Pfa, "rna_AUC": rna_AUC,
    "gid_Pd": gid_Pd, "gid_Pfa": gid_Pfa, "gid_AUC": gid_AUC,
    "pride_Pd": pride_Pd, "pride_Pfa": pride_Pfa, "pride_AUC": pride_AUC,
    "glrt_Pd": glrt_Pd, "glrt_Pfa": glrt_Pfa, "glrt_AUC": glrt_AUC,
    "wcfcpsc_Pd": wcfcpsc_Pd, "wcfcpsc_Pfa": wcfcpsc_Pfa, "wcfcpsc_AUC": wcfcpsc_AUC } 
# print(mdic)
file = result_directory + scenary + ".mat"
scipy.savemat(file, mdic)
