### load and format data
# Taken from from Boswell et al.

# packages 
import scipy.io as sio #
import tensorflow as tf #
import tensorflow_addons as tfa
import numpy as np #
import scipy.signal
import keras
import tensorflow.keras as keras
from keras.models import Sequential
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout
import tensorflow.python.keras.backend as K
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt #
#%matplotlib inline
import csv
import scipy.stats
import os
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import KFold   #for k fold cross validation

#for hyperparameter optimization
#from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.datasets import mnist
from keras.utils import np_utils

# Flag if 3D, frontal, or sagittal
modeltype = '3D' # Options are '3D', 'frontal', or 'sagittal'
outputType = 'both' # Options are 'early', 'late', 'both', 'three', 'max', and 'all' for peak MCF
    
# Configure to use CPU or GPU (we are using CPU)
config = tf.compat.v1.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 0})
session = tf.compat.v1.Session(config=config)
K.set_session(session)

### Define fcns for later in script  
# Taken from Boswell et al., modified to say "MCF"

def r2_numpy(data,labels,model):
    y_pred2 = model.predict(data)
    mse = np.mean(np.square(y_pred2-labels))
    r2 = np.square(np.corrcoef(labels.T,y_pred2.T)[0,1])
    mae = np.mean(np.abs(y_pred2-labels))
    return r2,mse,mae

def PredictMCF(model,inputData):
    predictedMCF = model.predict(inputData[range(inputData.shape[0]),:])
    return predictedMCF

def PlotMCFpredictions(trueMCF,predictedMCF):
    # Plot predicted and true peaks vs. step
    plt.figure()
    truePlot = plt.plot(trueMCF)
    predPlot = plt.plot(predictedMCF)
    plt.ylabel('MCF Peak Comparison')
    plt.xlabel('Step')
    plt.legend(('True','Predicted'),loc=4);

    # Plot predicted vs. true peaks
    plt.figure()
    ax = plt.plot(trueMCF,predictedMCF,'.',color=(45/255, 107/255, 179/255),alpha=0.05)
    plt.axis('equal')
    plt.ylabel('Predicted MCF')
    plt.xlabel('True MCF')
    plt.ylim(1,4)
    plt.xlim(1,4)
    plt.plot([-1,4],[-1,4],'k')
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
def PlotTrainingCurves(trainResults,devResults,epochCount):
    # Plot training curves
    lossPlt = plt.plot(np.arange(1,epochCount+1),train_loss[range(epochCount)])
    DevlossPlt = plt.plot(np.arange(1,epochCount+1),dev_loss[range(epochCount)])

    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch Number');
    plt.legend(('Training','Dev'))

    plt.figure(2)
    r2Plt = plt.plot(np.arange(1,epochCount+1),train_r2[range(epochCount)])
    devr2Plt = plt.plot(np.arange(1,epochCount+1),dev_r2[range(epochCount)])
    plt.ylim([.2, 1])
    plt.ylabel('r^2')
    plt.xlabel('Epoch Number');
    plt.legend(('Training','Dev'))
    
if modeltype not in ['3D', 'frontal', 'sagittal']:
    raise ValueError("Error: Options are '3D' 'frontal' or 'sagittal'.")

### load input data

# Load input data
inputData = sio.loadmat("Data\inputData.mat")

# Load input data (X)
ik_input = inputData["ik"] # inverse kinematics (101, 31, 7779)
time_input = inputData["time"] # time (101, 1, 7779)
leg = inputData["leg"].T  # stance leg (per step) (7779,1)
subject = inputData["subject"].T  # subject number (per step) (7779,1)

# Load output data (Y)
MCF = inputData["MCF"] # MCF over time (101, 7779)
peakMCF_early = inputData["peakMCF_early"].T # early stance peak (SORT OF NOT WORKING RN)
peakMCF_late = inputData["peakMCF_late"].T # late stance peak (SORT OF NOT WORKING RN)
minMCF = inputData["minMCF"].T # mid stance valley
peakMCF = inputData["peakMCF"]

# Print output dimensions
print("Inverse kinematics: " + str(ik_input.shape))
print("Time: " + str(time_input.shape))
print("Stance leg: " + str(leg.shape))
print("Subject number: " + str(subject.shape))
print("Medial contact force: " + str(MCF.shape))
print("Peak MCF: " + str(peakMCF.shape))
print("Early-stance MCF peak: " + str(peakMCF_early.shape))
print("Late-stance MCF peak: " + str(peakMCF_late.shape))
print("Mid-stance MCF valley: " + str(minMCF.shape))

### format input data
# Leg dimensions (nSamples, 101, 1)
legBin = np.expand_dims(np.tile(leg,(1,101)),axis=2)
print("Leg: " + str(legBin.shape))

# Adjust joint angles to correct dimensions (nSamples, nTimesteps, nFeatures)
angles = np.transpose(ik_input, axes=[2, 0, 1])
print("Joint angles: " + str(angles.shape))

# Time dimensions (nSamples, 1, 1) - DON'T END UP USING
time = np.expand_dims(np.transpose(time_input), axis = 2)
print("Time: " + str(time.shape))

# Concatenate legBin with angles
inputMat = np.concatenate((angles, legBin), axis = 2)

# Resample inputMat (nTimesteps = 16, down from 101)
inputMat = scipy.signal.resample(inputMat, 16, axis = 1)

# Use positions from first half of stance
#inputMat = inputMat[:,0:50,:]
print("Input shape: " + str(inputMat.shape))

### format output data
if outputType == 'all':
    output = np.expand_dims(MCF.T, axis = 2)
    output = scipy.signal.resample(output, 32, axis = 1)
    print("Output shape is: " + str(output.shape))
    
elif outputType == 'three':
    output = np.concatenate([peakMCF_early, minMCF], axis = 1)
    output = np.expand_dims(np.concatenate([output, peakMCF_late], axis = 1),axis = 2)
    
    # plot both peaks and valley
    plt.figure()
    plt.plot(output[:,0,0]);
    plt.ylabel("Early-stance peak MCF (BW)");
    plt.xlabel("Step");
    plt.figure()
    plt.plot(output[:,1,0]);
    plt.ylabel("Mid-stance minimum MCF (BW)");
    plt.xlabel("Step");
    plt.figure()
    plt.plot(output[:,2,0]);
    plt.ylabel("Late-stance peak MCF (BW)");
    plt.xlabel("Step");

elif outputType == 'both':
    # next step: multiple outputs (e.g. early and late-stance peaks in MCF)
    output = np.expand_dims(np.concatenate([peakMCF_early, peakMCF_late], axis = 1),axis = 2)
    print("Output shape is: " + str(output.shape))

    # plot both peaks
    plt.figure(1)
    plt.plot(output[:,0,0]);
    plt.ylabel("Early-stance peak MCF (BW)");
    plt.xlabel("Step");
    plt.figure(2)
    plt.plot(output[:,1,0]);
    plt.ylabel("Late-stance peak MCF (BW)");
    plt.xlabel("Step");

else:
    if outputType == 'early':
        # Reshape the output (nSamples, 1, 1)
        output = np.expand_dims(peakMCF_early,axis=2)
    
    elif outputType == 'late':
        output = np.expand_dims(peakMCF_early,axis=2)
        
    elif outputType == 'max':
        output = np.expand_dims(peakMCF,axis=2)

    print("Output shape is " + str(output.shape))
    # Plot output data
    plt.plot(output[:,0,0]);
    plt.ylabel("Peak MCF (BW)");
    plt.xlabel("Step");


### divide into train dev test
# Set the seed so it is reproducible
np.random.seed(1)
nSubjects = len(np.unique(subject)) # 68 subjects
subject_shuffle = np.unique(subject)
np.random.shuffle(subject_shuffle)

# 80-10-10 split (54-7-7 subjects)
train, dev, test = np.split(subject_shuffle, [int(0.8*len(subject_shuffle)), int(0.9*len(subject_shuffle))])
print("Train: " + str(len(train)) + " subjects")
print("Dev: " + str(len(dev)) + " subjects")
print("Test: " + str(len(test)) + " subjects")

# Find step indicies for each subject in each set (taken from Boswell et al., 2021)
trainInds = np.array(0)
for i in train:
    trainInds = np.append(trainInds,np.argwhere(subject==i)[:,0])
trainInds = trainInds[1:]
    
devInds = np.array(0)
for i in dev:
    devInds = np.append(devInds,np.argwhere(subject==i)[:,0])
devInds = devInds[1:]

testInds = np.array(0)
for i in test:
    testInds = np.append(testInds,np.argwhere(subject==i)[:,0])
testInds = testInds[1:]

# Build training, development, and test inputs and labels (taken from Boswell et al., 2021)
trainInput_full = inputMat[trainInds,:,:]
trainInput_full = trainInput_full.reshape((trainInput_full.shape[0],-1)) # flatten
trainLabels = output[trainInds,:,0]

devInput_full = inputMat[devInds,:,:]
devInput_full = devInput_full.reshape((devInput_full.shape[0],-1)) # flatten
devLabels = output[devInds,:,0]

testInput_full = inputMat[testInds,:,:]
testInput_full = testInput_full.reshape((testInput_full.shape[0],-1))
testLabels = output[testInds,:,0]

### remove redundant leg
# Extract indices of leg (every 32nd index, leave first one)
inputIdx = np.delete(np.arange(0,trainInput_full.shape[1]), np.arange(63, trainInput_full.shape[1], 32))

# Could also do some sort of input feature selection here if we wanted to!

# Remove additional leg input features
trainInput = trainInput_full[:,inputIdx]
devInput = devInput_full[:,inputIdx]
testInput = testInput_full[:,inputIdx]

print("Train input: " + str(trainInput.shape))
print("Dev input: " + str(devInput.shape))
print("Test input: " + str(testInput.shape))

### hyperparameter optimization

def data():
    """
    Data providing function:
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """

    modeltype = '3D' # Options are '3D', 'frontal', or 'sagittal'
    outputType = 'both' # Options are 'early', 'late', 'both', 'three', 'max', and 'all' for peak MCF
    
    # Load input data
    inputData = sio.loadmat("Data\inputData.mat")

    # Load input data (X)
    ik_input = inputData["ik"] # inverse kinematics (101, 31, 7779)
    time_input = inputData["time"] # time (101, 1, 7779)
    leg = inputData["leg"].T  # stance leg (per step) (7779,1)
    subject = inputData["subject"].T  # subject number (per step) (7779,1)

    # Load output data (Y)
    MCF = inputData["MCF"] # MCF over time (101, 7779)
    peakMCF_early = inputData["peakMCF_early"].T # early stance peak (SORT OF NOT WORKING RN)
    peakMCF_late = inputData["peakMCF_late"].T # late stance peak (SORT OF NOT WORKING RN)
    minMCF = inputData["minMCF"].T # mid stance valley
    peakMCF = inputData["peakMCF"]

    # Print output dimensions
    print("Inverse kinematics: " + str(ik_input.shape))
    print("Time: " + str(time_input.shape))
    print("Stance leg: " + str(leg.shape))
    print("Subject number: " + str(subject.shape))
    print("Medial contact force: " + str(MCF.shape))
    print("Peak MCF: " + str(peakMCF.shape))
    print("Early-stance MCF peak: " + str(peakMCF_early.shape))
    print("Late-stance MCF peak: " + str(peakMCF_late.shape))
    print("Mid-stance MCF valley: " + str(minMCF.shape))

    ### format input data
    # Leg dimensions (nSamples, 101, 1)
    legBin = np.expand_dims(np.tile(leg,(1,101)),axis=2)
    print("Leg: " + str(legBin.shape))

    # Adjust joint angles to correct dimensions (nSamples, nTimesteps, nFeatures)
    angles = np.transpose(ik_input, axes=[2, 0, 1])
    print("Joint angles: " + str(angles.shape))

    # Time dimensions (nSamples, 1, 1) - DON'T END UP USING
    time = np.expand_dims(np.transpose(time_input), axis = 2)
    print("Time: " + str(time.shape))

    # Concatenate legBin with angles
    inputMat = np.concatenate((angles, legBin), axis = 2)

    # Resample inputMat (nTimesteps = 16, down from 101)
    inputMat = scipy.signal.resample(inputMat, 16, axis = 1)

    # Use positions from first half of stance
    #inputMat = inputMat[:,0:50,:]
    print("Input shape: " + str(inputMat.shape))

    ### format output data
    if outputType == 'all':
        output = np.expand_dims(MCF.T, axis = 2)
        output = scipy.signal.resample(output, 32, axis = 1)
        print("Output shape is: " + str(output.shape))
        
    elif outputType == 'three':
        output = np.concatenate([peakMCF_early, minMCF], axis = 1)
        output = np.expand_dims(np.concatenate([output, peakMCF_late], axis = 1),axis = 2)
        
        # plot both peaks and valley
        plt.figure()
        plt.plot(output[:,0,0]);
        plt.ylabel("Early-stance peak MCF (BW)");
        plt.xlabel("Step");
        plt.figure()
        plt.plot(output[:,1,0]);
        plt.ylabel("Mid-stance minimum MCF (BW)");
        plt.xlabel("Step");
        plt.figure()
        plt.plot(output[:,2,0]);
        plt.ylabel("Late-stance peak MCF (BW)");
        plt.xlabel("Step");

    elif outputType == 'both':
        # next step: multiple outputs (e.g. early and late-stance peaks in MCF)
        output = np.expand_dims(np.concatenate([peakMCF_early, peakMCF_late], axis = 1),axis = 2)
        print("Output shape is: " + str(output.shape))

        # plot both peaks
        plt.figure(1)
        plt.plot(output[:,0,0]);
        plt.ylabel("Early-stance peak MCF (BW)");
        plt.xlabel("Step");
        plt.figure(2)
        plt.plot(output[:,1,0]);
        plt.ylabel("Late-stance peak MCF (BW)");
        plt.xlabel("Step");

    else:
        if outputType == 'early':
            # Reshape the output (nSamples, 1, 1)
            output = np.expand_dims(peakMCF_early,axis=2)
        
        elif outputType == 'late':
            output = np.expand_dims(peakMCF_early,axis=2)
            
        elif outputType == 'max':
            output = np.expand_dims(peakMCF,axis=2)

    print("Output shape is " + str(output.shape))
    # Plot output data
    plt.plot(output[:,0,0]);
    plt.ylabel("Peak MCF (BW)");
    plt.xlabel("Step");


    ### divide into train dev test
    # Set the seed so it is reproducible
    np.random.seed(1)
    nSubjects = len(np.unique(subject)) # 68 subjects
    subject_shuffle = np.unique(subject)
    np.random.shuffle(subject_shuffle)

    # 80-10-10 split (54-7-7 subjects)
    train, dev, test = np.split(subject_shuffle, [int(0.8*len(subject_shuffle)), int(0.9*len(subject_shuffle))])
    print("Train: " + str(len(train)) + " subjects")
    print("Dev: " + str(len(dev)) + " subjects")
    print("Test: " + str(len(test)) + " subjects")

    # Find step indicies for each subject in each set (taken from Boswell et al., 2021)
    trainInds = np.array(0)
    for i in train:
        trainInds = np.append(trainInds,np.argwhere(subject==i)[:,0])
    trainInds = trainInds[1:]
        
    devInds = np.array(0)
    for i in dev:
        devInds = np.append(devInds,np.argwhere(subject==i)[:,0])
    devInds = devInds[1:]

    testInds = np.array(0)
    for i in test:
        testInds = np.append(testInds,np.argwhere(subject==i)[:,0])
    testInds = testInds[1:]

    # Build training, development, and test inputs and labels (taken from Boswell et al., 2021)
    trainInput_full = inputMat[trainInds,:,:]
    trainInput_full = trainInput_full.reshape((trainInput_full.shape[0],-1)) # flatten
    trainLabels = output[trainInds,:,0]

    devInput_full = inputMat[devInds,:,:]
    devInput_full = devInput_full.reshape((devInput_full.shape[0],-1)) # flatten
    devLabels = output[devInds,:,0]

    testInput_full = inputMat[testInds,:,:]
    testInput_full = testInput_full.reshape((testInput_full.shape[0],-1))
    testLabels = output[testInds,:,0]

    ### remove redundant leg
    # Extract indices of leg (every 32nd index, leave first one)
    inputIdx = np.delete(np.arange(0,trainInput_full.shape[1]), np.arange(63, trainInput_full.shape[1], 32))

    # Could also do some sort of input feature selection here if we wanted to!

    # Remove additional leg input features
    trainInput = trainInput_full[:,inputIdx]
    devInput = devInput_full[:,inputIdx]
    testInput = testInput_full[:,inputIdx]

    x_train = trainInput
    y_train = trainLabels
    x_test = devInput
    y_test = devLabels  

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
        Double curly brackets indicate parameters to optimize with hyperas
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    # Model definition / hyperparameters space definition / fit / eval

    loss = 'mse'
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    np.random.seed(2)
    tf.compat.v1.set_random_seed(2)

    model = Sequential()
    kernel_regularizer= tf.keras.regularizers.L2({{uniform(0, 1)}})
    bias_regularizer= tf.keras.regularizers.l2({{uniform(0, 1)}}) 

    # batch norm layer here
    
    model.add(Dense(800,input_shape = (input_dim,), kernel_initializer=glorot_normal(seed=None) , activation='relu'))
    
    nHiddenLayers = {{choice([1,2,3,4,5,6,7,8,9,10])}}   #pick random int between 1-10
    nHiddenUnits = {{choice([np.arange(5,1000,5)])}}   #pick random int between 1-1000 with step of 10

    for i in range(nHiddenLayers-1):
        
        #model.add(Dense(nHiddenUnits, kernel_initializer=glorot_normal(seed=None) , activation='relu', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
        
        model.add(Dense(nHiddenUnits, kernel_initializer=glorot_normal(seed=None) , activation='relu', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
        model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense(output_dim,kernel_initializer=glorot_normal(seed=None),activation='linear'))
    model.add(Dropout({{uniform(0, 1)}}))
    # rmse = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss=loss,optimizer='adam', metrics=['mse'])

    #if not improving
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = "min", patience=20, restore_best_weights = True)

    # fit model on training data
    history = myModel.fit(trainInput, trainLabels, validation_data=(devInput, devLabels), epochs = 100, batch_size={{choice([32, 64, 128])}}, callbacks = [callback], verbose = 2)


    #loss = mse loss of test and dev sets
    loss = history.history['loss'] + history.history['dev_loss']
    print('best loss: ', loss)

    return {'loss': loss, 'status': STATUS_OK, 'model': create_model}

x_train, y_train, x_test, y_test = data()

best_run = optim.minimize(model=create_model,
                          data=data,
                          algo=tpe.suggest,
                          max_evals=10,
                          trials=Trials()
                          )