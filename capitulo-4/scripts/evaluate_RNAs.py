import sys, os

# Set path to find custom modules
sys.path.append( os.path.dirname(os.path.dirname(  os.path.abspath(__file__) ) ) )

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from dataset import dataset
from dataset.noise import noise as noise
from detectors import wcfcpsc, glrt
from chart import chart

###################################################################### RNA imports
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.densenet import DenseNet201
from keras.applications.resnet_v2 import ResNet152V2

relative_work_directory = "LOCAL_FILES"
#################################################################################################### Create work directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
absolute_work_directory = os.path.join(parent_dir, relative_work_directory) 

dataset_directory = absolute_work_directory + "/datasets/"
model_directory = absolute_work_directory + "/models/"
result_directory = absolute_work_directory + "/result/evaluate_RNAs/"

if not os.path.exists(model_directory):
    os.makedirs(model_directory, exist_ok = True)
if not os.path.exists(result_directory):
    os.makedirs(result_directory, exist_ok = True)
#################################################################################################### Load Args
import sys

N_ARGS_EXPECTED = 5 + 1 # +1 for the script name

# total arguments
n = len(sys.argv)
if n != N_ARGS_EXPECTED:
    raise Exception("Expected %d arguments, but got %d" % (N_ARGS_EXPECTED, n))
 
keras_model_str = sys.argv[1]
model_name = sys.argv[2]
add3Channel = bool(sys.argv[3])
preCalc = int(sys.argv[4])
resize = int(sys.argv[5])

print("keras_model: ", keras_model_str)
print("model_name: ", model_name)
print("add3Channel: ", add3Channel)
print("preCalc: ", preCalc)
print("resize: ", resize)

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.densenet import DenseNet201
from keras.applications.resnet_v2 import ResNet152V2

if keras_model_str == "VGG16":
    keras_model = VGG16
elif keras_model_str == "InceptionV3":
    keras_model = InceptionV3
elif keras_model_str == "EfficientNetB7":
    keras_model = EfficientNetB7
elif keras_model_str == "DenseNet201":
    keras_model = DenseNet201
elif keras_model_str == "ResNet152V2":
    keras_model = ResNet152V2
else:
    raise Exception("Invalid keras_model_str: %s" % keras_model_str)

#################################################################################################### Configuration
# add3Channel = False

# keras_model, model_name, add3Channel, preCalc, resize  = DenseNet201 , "DenseNet201", False, 0, 6
# keras_model, model_name, add3Channel, preCalc, resize  = ResNet152V2 , "ResNet152V2", False, 0, 6
# keras_model, model_name, add3Channel, preCalc, resize  = InceptionV3 , "InceptionV31", False, 0,  3
# keras_model, model_name, add3Channel, preCalc, resize  = EfficientNetB7 , "EfficientNetB7", False, 0,  6
# keras_model, model_name, add3Channel, preCalc, resize  = VGG16 , "VGG16", False, 0, 6

# keras_model, model_name, add3Channel, preCalc, resize  = DenseNet201 , "DenseNet201_FFT", False, 1, 6
# keras_model, model_name, add3Channel, preCalc, resize  = ResNet152V2 , "ResNet152V2_FFT", False, 1, 6
# keras_model, model_name, add3Channel, preCalc, resize  = InceptionV3 , "InceptionV31_FFT", False, 1, 3
# keras_model, model_name, add3Channel, preCalc, resize  = EfficientNetB7 , "EfficientNetB7_FFT", False, 1, 6
# keras_model, model_name, add3Channel, preCalc, resize  = VGG16 , "VGG16_FFT", True, 1, 6

# keras_model, model_name, add3Channel, preCalc, resize  = DenseNet201 , "DenseNet201_SCM", False, 2, 4
# keras_model, model_name, add3Channel, preCalc, resize  = ResNet152V2 , "ResNet152V2_SCM", False, 2, 4
# keras_model, model_name, add3Channel, preCalc, resize  = InceptionV3 , "InceptionV31_SCM", False, 2, 5
# keras_model, model_name, add3Channel, preCalc, resize  = EfficientNetB7 , "EfficientNetB7_SCM", False, 2, 4
# keras_model, model_name, add3Channel, preCalc, resize  = VGG16 , "VGG16_SCM", True, 2, 4

retrain_model = False
reset_model = False

#################################################################################################### Load Dataset
X, Y = dataset.loadDataset(filename= dataset_directory + "selective_n(160)_m(6)_SNR(-10)_Sigma2avg(1)_rhoN(0.00)_PUsignal(1)_SPS(4)_shadowing(True)_p(4)_noise(non-uniform).mat")

if preCalc == 0:
    p1 = X
elif preCalc == 1:
    p1 = dataset.calcFFT(X)
elif preCalc == 2:
    p1 = dataset.calcSCM(X)
else:
    sys.exit("Invalid preCalc value")

_pX, pY, input_shape = dataset.cplxToColum_and_hotEncoded(p1,Y)

if resize == 1:
    pX = dataset.reshapeDataset(_pX, 36,240)
    input_shape = (36,240,2)
elif resize == 2:
    pX = dataset.reshapeDataset(_pX, 75,240)
    input_shape = (75,240,2)
elif resize == 3:
    pX = dataset.reshapeDataset(_pX, 75,160)
    input_shape = (75,160,2)
elif resize == 4:
    pX = dataset.reshapeDataset(_pX, 36,36)
    input_shape = (36,36,2)
elif resize == 5:
    pX = dataset.reshapeDataset(_pX, 75,75)
    input_shape = (75,75,2)
elif resize == 6:
    pX = dataset.reshapeDataset(_pX, 36,160)
    input_shape = (36,160,2)
else:
    sys.exit("Invalid resize value")

if add3Channel:
    pX = dataset.add3Channel(pX)
    input_shape = (input_shape[0],input_shape[1],3)

x_train, y_train, x_val, y_val, x_test, y_test = dataset.splitDataset(pX, pY)

#################################################################################################### Train Model
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
    # opt = Adam(lr=0.00005)
    # model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
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
    checkpoint = ModelCheckpoint(model_directory + model_name + ".h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
    #                                 patience=2, 
    #                                 verbose=1, 
    #                                 factor=0.1, 
    #                                 min_lr=1e-6)
    # hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=32, callbacks=[checkpoint, es, reduce_lr])
    hist = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), batch_size=32, callbacks=[checkpoint, es])
    # chart.plotHistory(hist)

    # Load best model
    model = load_model(model_directory + model_name + ".h5")

# _, accuracy = model.evaluate(x_test, y_test, verbose=1)
# print("accuracy : ", accuracy)
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
Pfa, Pd, AUC = wcfcpsc.evaluate_dataset(X, Y, nSubBands=5, Npt=40)
c.add_plot(Pfa, Pd, AUC, label="WCFCPSC", style='b-v')
wcfcpsc_Pfa, wcfcpsc_Pd, wcfcpsc_AUC = Pfa, Pd, AUC

Pfa, Pd, AUC = glrt.evaluate_dataset(X, Y, Npt=40)
c.add_plot(Pfa, Pd, AUC, label="GLRT", style='r-v')
glrt_Pfa, glrt_Pd, glrt_AUC = Pfa, Pd, AUC

c.save(result_directory + model_name )
# c.show()

## Save pd e pfa 
import scipy.io as scipy
if retrain_model or reset_model:
    mdic = {"rna_Pd": rna_Pd, "rna_Pfa": rna_Pfa, "rna_AUC": rna_AUC , "acc": hist.history['accuracy'], "val_acc": hist.history['val_accuracy'], "loss": hist.history['loss'], "val_loss": hist.history['val_loss'] }
else:
    mdic = {"rna_Pd": rna_Pd, "rna_Pfa": rna_Pfa, "rna_AUC": rna_AUC } 
# print(mdic)
file = result_directory + model_name + ".mat"
scipy.savemat(file, mdic)
