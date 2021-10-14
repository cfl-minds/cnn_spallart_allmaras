
"""
-------------------------------------------------------------------------------------------------
Supporting:     Robust Deep Learning For Emulating Turbulent Viscosities - Physics of Fluids
URL:            https://arxiv.org/abs/2107.11235
Author:         Aakash Patil, Jonathan Viquerat, George El Haber, Elie Hachem                             
Year:           September, 2021                                                
-------------------------------------------------------------------------------------------------
"""
import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.keras import Input, models
from custom_layers import *


def make_patchedDataset(inputDataset, window=(50,50), dwindow=(25,25),flip=False):
      win_h, win_w = window
      dwin_h, dwin_w = dwindow
      ksize_rows,ksize_cols = win_h, win_w #50,50
      strides_rows, strides_cols = dwin_h, dwin_w #25, 25
      ksizes = [1, ksize_rows, ksize_cols, 1] 
      strides = [1, strides_rows, strides_cols, 1]
      rates = [1, 1, 1, 1] 
      padding='VALID' # 'SAME'

      print("Input Dataset.shape is ", inputDataset.shape)
      #flip dataset
      if flip:
            print("Flipping will now expand dataset 2 times [NOT TESTED] ")
            #flippedDataSet = tf.image.flip_left_right(inputDataset)
            inputDataset = tf.concat([inputDataset, tf.image.flip_left_right(inputDataset)],axis=0)
            #inputDataset = tf.concat([inputDataset, tf.image.flip_up_down(inputDataset)],axis=0)
            print("Dataset shape after flipping is ", inputDataset.shape)

      nch = inputDataset.shape[-1]
      nbatches = inputDataset.shape[0]

      dataset_patches = tf.image.extract_patches(inputDataset, ksizes, strides, rates, padding)
      nx_slices, ny_slices = dataset_patches.shape[1] , dataset_patches.shape[2]
      patches_list = []
      for row in range(nx_slices):
            for column in range(ny_slices):
                  thispatch = tf.reshape(dataset_patches[:,row,column,:], [nbatches,ksize_rows, ksize_cols, nch])
                  patches_list.append(thispatch)

      dataSet_expanded = patches_list[0]
      for i in range (1,len(patches_list)):
            dataSet_expanded = tf.concat([dataSet_expanded,patches_list[i]],axis=0)
      
      print("Window: ", window, " Window-centers: ", dwindow)
      print("Suggestion: plt.subplots(ny, ny)" ,nx_slices,ny_slices )
      print("Output Dataset.shape is ", dataSet_expanded.shape)
      return dataSet_expanded;
#example: import numpy as np; expandedSample = make_patchedDataset(np.zeros((25,198,159,2)), window=(50,50), dwindow=(25,25),flip=True)



def make_model(input_shape, activ_type='Prelu', superScale=1, gen_mode='GAN',outChannels=None,apply_BC=False,latShapeVal=(4,4)):
    print("#"*35)
    if len(input_shape)>3:
            print("3D Network")
            nx,ny,nz = input_shape[0], input_shape[1], input_shape[2]
            #from autoEnc_GetInfo import autoEnc_getEncMax_3D
            zComp,outEncShape = autoEnc_getEncMax_3D(shape=(nx,ny,nz),latShape = (latShapeVal[0],latShapeVal[1],latShapeVal[2]))
            if outEncShape !=  (input_shape[0], input_shape[1], input_shape[2]) :
                  flag_ReshapeOutput = True    
    elif len(input_shape)==3:
            print("2D Network")
            nx,ny = input_shape[0], input_shape[1]
            #from autoEnc_GetInfo import autoEnc_getEncMax_2D
            zComp,outEncShape = autoEnc_getEncMax_2D(shape=(nx,ny),latShape = (latShapeVal[0],latShapeVal[1])) #(4,4))
            if outEncShape !=  (input_shape[0], input_shape[1]) :
                  flag_ReshapeOutput = True 
    else:
            print("UNKNOWN Dimensional Network")
    ##input layer
    input_layer = Input(shape = input_shape)
    input_channels = input_shape[-1] 
    numfilter_base = 4 # or = input_channels or factor of 2^N

    netList = []
    fliterList = []
    ##Encoder
    for i in range(int(zComp)-1):
          ##m-block1
          filters = 2**(i+5)
          if i == 0:
                  net = block_conv(input_layer, filters, alpha=3, beta=1, mode='No_BN',  apply_BC=apply_BC)
                  netList.append(net)
                  fliterList.append(filters)
          else:
                  net = block_conv(net, filters, alpha=3, beta=1, mode='No_BN',  apply_BC=apply_BC)
          net = block_conv(net,         filters, alpha=3, beta=2, mode=gen_mode, apply_BC=False)  
          net = activationLayer(net, activ_type)
          fliterList.append(filters)
          netList.append(net)

    ##Latent Space
    netList = netList [::-1]
    fliterList = fliterList[::-1]
 
    if superScale == 2 :
          zComp = zComp + 1 
          fliterList.append(fliterList[-1]//2)
          if superScale == 4 :
                    zComp = zComp + 1 
                    fliterList.append(fliterList[-1]//2)
    ##Decoder
    for i in range(1,int(zComp)):
          ##deconv1
          filters = fliterList[i]
          net = block_conv(net, filters, alpha=3, beta=1, mode='No_BN',apply_BC=apply_BC)
          net = block_deconv(net, filters, method='ConvT',apply_BC=False,alpha=3)   
          net = activationLayer(net, activ_type)
          try:  
            if (net.shape[1],net.shape[2]) != (netList[i].shape[1],netList[i].shape[2]) :
                  print("Applying layerResize2D as net.shape=",net.shape, " expected:",netList[i].shape )
                  net = layerResize2D(newShape=(netList[i].shape[1], netList[i].shape[2]), newScale=1, usemethod='bilinear')(net)      
          except:
            print("Probably superScale > 1. Not using layerResize2D")  

    if superScale == 1 :
        if (net.shape[1],net.shape[2]) !=  (input_shape[0], input_shape[1]) :
            print("Unequal shapes. Using layerResize2D")
            net = layerResize2D(newShape=(input_shape[0], input_shape[1]), newScale=1, usemethod='bilinear')(net)
            
    if superScale > 1 :
        if (net.shape[1],net.shape[2]) !=  (input_shape[0]*superScale, input_shape[1]*superScale) :
            print("Unequal shapes. Using layerResize2D or 3D")
            try:
                net = layerResize2D(newShape=(input_shape[0]*superScale, input_shape[1]*superScale), newScale=1, usemethod='bilinear')(net)
                
            except:
                net = layerResize3D(newShape=(input_shape[0]*superScale, input_shape[1]*superScale, input_shape[2]*superScale), newScale=1, 
usemethod='bilinear')(net)
    #output channels 
    if outChannels==None:
       outChannels = input_channels
      
    ##conv
    net = block_conv(net, filters=outChannels, alpha=1, beta=1, mode='No_BN',apply_BC=apply_BC) #alpha=9
    ##make model
    model = models.Model(inputs=input_layer, outputs=net, name = 'model_'+gen_mode) 
    print("#"*35)
    #print (model.summary())
    return model;

#generator = make_model(input_shape=data_LR[0].shape, activ_type='ReLU', res_blocks=12, superScale=1, gen_mode='GAN',outChannels=data_HR[0].shape[-1],apply_BC=False,latShapeVal=(13,13))

def autoEnc_getEncMax_1D(nx = 50,latx = 5):
      num = nx
      limnum = latx
      maxIter = 20
      c = 0 
      c = num//2 
      print("INP layer    : ",num) 
      print("After Pool  1 : ",c) 
      for i in range(maxIter): 
          c = c//2 
          print("After Pool ",i+2,": ", c) 
          if c <= limnum: 
              cLast = c 
              break 
      print("LAT SPACE    : ",cLast) 
      encMax = i+2 
      print("Max number of encoding Conv:",encMax) 
       
      co = 0 
      co = cLast*2 
      print("After UpSamp  1 : ",co) 
      for i in range(encMax-1): 
          co = co*2 
          print("After UpSamp ",i+2,": ",co) 
      print    ("OUT layer      : ", co)     
      decMax = i+2 
      print("Max number of decoding Conv:", decMax)  
      if encMax == decMax : 
          print("Returning encMax Value") 
      else: 
          print("Error ! Please check shapes")      
      return encMax, co;

def autoEnc_getEncMax_2D(shape=(50,50),latShape = (5,5)):
      nx, ny = shape[0], shape[1]
      latx, laty = latShape[0], latShape[1]
      encMax_x,out_x  = autoEnc_getEncMax_1D(nx,latx)
      encMax_y,out_y  = autoEnc_getEncMax_1D(ny,laty)
      if encMax_x  == encMax_y  : 
          print("##########  Returning encMax Effective Value #####") 
      else: 
          print("Error ! Please check shapes")  
      return encMax_x,(out_x,out_y)


def autoEnc_getEncMax_3D(shape=(50,50,50),latShape = (5,5,5)):
      nx, ny, nz = shape[0], shape[1], shape[2]
      latx, laty, latz = latShape[0], latShape[1], latShape[2]
      encMax_x,out_x  = autoEnc_getEncMax_1D(nx,latx)
      encMax_y,out_y  = autoEnc_getEncMax_1D(ny,laty)
      encMax_z,out_z  = autoEnc_getEncMax_1D(nz,latz)
      if (encMax_x  == encMax_y) and  (encMax_x  == encMax_z) : 
          print("##########  Returning encMax Effective Value #####") 
      else: 
          print("Error ! Please check shapes")  
      return encMax_x, (out_x,out_y,out_z)



