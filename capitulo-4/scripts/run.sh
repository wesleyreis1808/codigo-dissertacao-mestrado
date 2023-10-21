
LOGS_DIR="../LOCAL_FILES/result/evaluate_RNAs/logs"
SCRIPT="evaluate_RNAs.py"

mkdir -p $LOGS_DIR
####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=DenseNet201
MODEL_NAME=DenseNet201
ADD_3_CHANNEL=False
PRE_CALC=0
RESIZE=6

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

# sed -i '/ETA:/d' $LOGS_DIR/$MODEL_NAME.log
####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=ResNet152V2
MODEL_NAME=ResNet152V2
ADD_3_CHANNEL=False
PRE_CALC=0
RESIZE=6

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=InceptionV3
MODEL_NAME=InceptionV31
ADD_3_CHANNEL=False
PRE_CALC=0
RESIZE=3

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=EfficientNetB7
MODEL_NAME=EfficientNetB7
ADD_3_CHANNEL=False
PRE_CALC=0
RESIZE=6

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FFT
####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=DenseNet201
MODEL_NAME=DenseNet201_FFT
ADD_3_CHANNEL=False
PRE_CALC=1
RESIZE=6

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=ResNet152V2
MODEL_NAME=ResNet152V2_FFT
ADD_3_CHANNEL=False
PRE_CALC=1
RESIZE=6

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=InceptionV3
MODEL_NAME=InceptionV31_FFT
ADD_3_CHANNEL=False
PRE_CALC=1
RESIZE=3

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=EfficientNetB7
MODEL_NAME=EfficientNetB7_FFT
ADD_3_CHANNEL=False
PRE_CALC=1
RESIZE=6

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SCM
####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=DenseNet201
MODEL_NAME=DenseNet201_SCM
ADD_3_CHANNEL=False
PRE_CALC=2
RESIZE=4

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=ResNet152V2
MODEL_NAME=ResNet152V2_SCM
ADD_3_CHANNEL=False
PRE_CALC=2
RESIZE=4

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=InceptionV3
MODEL_NAME=InceptionV31_SCM
ADD_3_CHANNEL=False
PRE_CALC=2
RESIZE=5

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 

####################################################################################################################
####################################################################################################################
# Params for training
KERAS_MODEL=EfficientNetB7
MODEL_NAME=EfficientNetB7_SCM
ADD_3_CHANNEL=False
PRE_CALC=2
RESIZE=4

touch $LOGS_DIR/$MODEL_NAME.log
start=`date +%s`

python -u $SCRIPT $KERAS_MODEL $MODEL_NAME $ADD_3_CHANNEL $PRE_CALC $RESIZE 2>&1 | tee $LOGS_DIR/$MODEL_NAME.log

end=`date +%s`
echo Execution time was `expr $end - $start` seconds.
echo "Execution time was `expr $end - $start` seconds." >> $LOGS_DIR/$MODEL_NAME.log 