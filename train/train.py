
"""
-------------------------------------------------------------------------------------------------
Supporting:     Robust Deep Learning For Emulating Turbulent Viscosities - Physics of Fluids
URL:            https://arxiv.org/abs/2107.11235
Author:         Aakash Patil, Jonathan Viquerat, George El Haber, Elie Hachem                             
Year:           September, 2021                                                
-------------------------------------------------------------------------------------------------
"""


import os
import sys
import glob
import numpy as np
import time
from tqdm import tqdm  
#### Load TF2.0.0
print("Initializing TensorFlow . . .")
import tensorflow as tf
print("Using TensorFlow ", tf.__version__)


from model_arch import make_patchedDataset
from custom_layers import *

################ GPU Settings #################

## Set which GPU to use out of two V100 Teslas
GPU_to_use = 0 #1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    useGPU = GPU_to_use
    try:
        tf.config.experimental.set_visible_devices(gpus[useGPU], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        tf.config.experimental.set_memory_growth(gpus[useGPU], True)
    except RuntimeError as e:
        print(" Visible devices must be set before GPUs have been initialized")
        print(e)
######################################################

############
BATCH_SIZE = 16   
EPOCHS = 10
trainingLossesFile= 'trainingLosses.txt'
trainedModelSavePath = './'  
saveAfter = 4    
############

###### Percentage of data to use for training - remaining goes for validation
trainValidRatio = 75 # First this % would be used for training 


datasetfile = '../transport/dataset.npy'
data = np.load(datasetfile,mmap_mode='r')
print("Loaded npy shape is: ", data.shape)


##Learn NuTilde from Velocity
#quantity we want to learn
#N, nx,ny, ch
data_HR = np.expand_dims(data[:,:,:,-1] , axis = -1) #SA NuTilde 
#quantity to learn from
data_LR = data[:,:,:,:2] #velocities 0&1 

print("data_HR.shape ", data_HR.shape) 
print("data_LR.shape ", data_LR.shape) 
snapshots = data_HR.shape[0]
#### Assert sample size
if data_HR.shape[0] != data_LR.shape[0]:
      print("Unequal Samples in LR and HR\n Bye !")
      sys.exit(); 
######################################################



## Domain Partitioning or Expansion
MakePatchedDataset = True
if MakePatchedDataset :
      window = (50,50)  ## patch size
      dwindow = (25,25)   ## patch stride
      data = np.append(data_HR,data_LR,axis=-1)
      data = make_patchedDataset(data, window, dwindow,flip=False)
      data_LR = data[...,:2].numpy()   
      data_HR = data[...,-1:].numpy()
      snapshots_expanded = data.shape[0]
      expansionsFromEachSnap = snapshots_expanded//snapshots
      snapshots = snapshots_expanded

#### Assert sample size
if data_HR.shape[0] != data_LR.shape[0]:
      print("After datasetExpansion: Unequal Samples in LR and HR\n Exiting !")
      sys.exit(); 



################## MAKE TF DATASET ##################
##amount of data to use for training
trainVal = int(snapshots*(trainValidRatio/100)) 

### Batch and shuffle the data if required
BUFFER_SIZE = trainVal  
train_dataset = tf.data.Dataset.from_tensor_slices((data_LR[:trainVal],data_HR[:trainVal])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = tf.data.Dataset.from_tensor_slices((data_LR[trainVal:],data_HR[trainVal:])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#model
from model_arch import make_model
generator = make_model(input_shape=data_LR[0].shape, activ_type='ReLU', superScale=1, gen_mode='GAN',outChannels=data_HR[0].shape[-1],apply_BC=False,latShapeVal=(data_LR[0].shape[0]/(2**6),data_LR[0].shape[0]/(2**6) )) 
print(generator.summary() )

print("Test Sample shape ", data_LR[:1,:,:,:].shape) 
generated_tensor = generator(data_LR[:1,:,:,:], training=False)
print("generated_tensor.shape", generated_tensor.shape)


generator_optimizer = tf.keras.optimizers.Adam(1e-4) #1e-3  
###################### TRAINING AND VALIDATION STEP ##########################
def train_step(data_LR,data_HR):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_velocity = generator(data_LR, training=True)
            gen_loss_min = tf.math.reduce_mean(tf.keras.losses.MSE(data_HR,generated_velocity) )
    gradients_of_generator = gen_tape.gradient(gen_loss_min, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss_min;

def validation_step(data_LR,data_HR):
            generated_velocity = generator(data_LR, training=False)
            gen_loss = tf.math.reduce_mean(tf.keras.losses.MSE(data_HR,generated_velocity) )
            return gen_loss;

####################################################################

def train(dataset, epochs, saveAfter, validation_dataset=None):

  gen_lossArr = []
  gen_lossArr_validation = [] 

  for epoch in tqdm(range(epochs),total=epochs,bar_format="Epochs: {percentage:.0f}%|{bar}{r_bar}"):
    start = time.time()  
    gen_lossFB = 0.000
    #c = 0
    totalBatches = len(list(dataset))
    print("Found totalBatches=",totalBatches)
    for data_batch in tqdm(dataset,total=totalBatches,bar_format="Batches: {percentage:.0f}%|{bar}{r_bar}"):
        gen_loss = train_step(data_batch[0],data_batch[1])
        gen_lossFB += gen_loss
        print("\nBatch gen_loss: ", gen_loss )

    gen_lossFB = gen_lossFB/totalBatches
    gen_lossArr.append(gen_lossFB)
    print("\nEpoch gen_loss: ", gen_lossFB )
    savearr=str(epoch+1) +'\t' + str(gen_lossFB) +'\t'  
    with open(trainingLossesFile, "a") as file_object:
        file_object.write("\n"+savearr)    

    gen_lossFB = 0.000
    totalBatchesV = len(list(validation_dataset))
    for vdata_batch in tqdm(validation_dataset,total=totalBatchesV,bar_format="Batches: {percentage:.0f}%|{bar}{r_bar}"):
        gen_loss = validation_step(vdata_batch[0],vdata_batch[1])
        gen_lossFB += gen_loss
        print("\nBatch gen_loss Validation: ", gen_loss )

    gen_lossFB = gen_lossFB/totalBatchesV
    gen_lossArr_validation.append(gen_lossFB) 

    print("\nEpoch gen_loss Validation: ", gen_lossFB )
    savearr=str(epoch+1) +'\t' + str(gen_lossFB) +'\t'  
    with open('Validation_'+trainingLossesFile, "a") as file_object:
        file_object.write("\n"+savearr)    

    if (epoch + 1) % saveAfter == 0:
      generator.save(trainedModelSavePath+'/'+'model_at_epoch_{:04d}.h5'.format(epoch+1))

    print ('\n Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    earlyStopSignal = Callback_EarlyStopping(gen_lossArr_validation, min_delta=0.01, patience=saveAfter,verbose=True)
    #earlyStopSignal = False #NO STOPPING EARLY
    if earlyStopSignal:
        print("Early stopping signal received at epoch=",epoch,"/",epochs)
        print("Breaking the training loop")
        break

   
  generator.save(trainedModelSavePath+'/'+'model_at_epoch_LastBest.h5')
  gen_lossArr = [ elem.numpy() for elem in gen_lossArr ]
  gen_lossArr_validation = [ elem.numpy() for elem in gen_lossArr_validation ]
  return gen_lossArr,gen_lossArr_validation  ;  
        



###################### START TRAINING LOOP #########################

print("#"*25)
print("Starting Training[here, training is gerund ;-)]")
gen_lossArr  = train(train_dataset, EPOCHS, saveAfter,validation_dataset)  


try:
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(np.arange(len(gen_lossArr[0])), gen_lossArr[0], '-b', label='Training')
    plt.plot(np.arange(len(gen_lossArr[1])), gen_lossArr[1], '-r', label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig('TrainingCurve.pdf')
except:
    print('Unable to plot. Do manual.')    

print("Training finished !")


###################### PREDICT/TEST ON ANOTHER DATASET #########################

loaded_model = tf.keras.models.load_model( 'model_at_epoch_LastBest.h5',custom_objects=custom_objects)

validation_datasetfile = '../transport/dataset.npy'
data = np.load(validation_datasetfile,mmap_mode='r')
print("Loaded npy shape is: ", data.shape)

##Learn NuTilde from Velocity
#quantity we want to learn
#N, nx,ny, ch
data_HR = np.expand_dims(data[:,:,:,-1] , axis = -1) #SA NuTilde 
#quantity to learn from
data_LR = data[:,:,:,:2] #velocities 0&1 

print("data_HR.shape ", data_HR.shape) 
print("data_LR.shape ", data_LR.shape) 
snapshots = data_HR.shape[0]
######################################################

## Domain Partitioning or Expansion
if MakePatchedDataset :
      data = np.append(data_HR,data_LR,axis=-1)
      data = make_patchedDataset(data, window, dwindow,flip=False)
      data_LR = data[...,:2].numpy()   
      data_HR = data[...,-1:].numpy()
      snapshots_expanded = data.shape[0]
      expansionsFromEachSnap = snapshots_expanded//snapshots
      snapshots = snapshots_expanded

data_HR_DL = loaded_model.predict( data_LR )

from scipy.stats import describe

print("Expected  : ", describe( data_HR.flatten() ) )
print("Predicted : ", describe( data_HR_DL.flatten() ) )
print("ABS Error : ", describe( (data_HR - data_HR_DL).flatten() ) )






