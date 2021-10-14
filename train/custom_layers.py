
"""
-------------------------------------------------------------------------------------------------
Supporting:     Robust Deep Learning For Emulating Turbulent Viscosities - Physics of Fluids
URL:            https://arxiv.org/abs/2107.11235
Author:         Aakash Patil, Jonathan Viquerat, George El Haber, Elie Hachem                             
Year:           September, 2021                                                
-------------------------------------------------------------------------------------------------
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class layerResize2D(layers.Layer):
      def __init__(self,newShape=None, newScale=2, usemethod='bilinear',**kwargs):
          super(layerResize2D, self).__init__(**kwargs)
          self.newShape = newShape
          self.newScale = newScale
          self.usemethod = usemethod
      def call(self, inputs):
          #print("layerInput.shape", inputs.get_shape())
          if self.newScale > 1 :
            newScale = int(self.newScale)
            dims = inputs.get_shape()
            newH, newW = dims[1]*newScale, dims[2]*newScale
          else:  
            newShape = self.newShape
            newH, newW = newShape[0], newShape[1] 
          #print("layer_resize newH,newW ", newH,newW )
          layerOutput = tf.image.resize(images=inputs, size=(newH, newW), method=self.usemethod, preserve_aspect_ratio=False, antialias=False)
          #print("layerOutput.shape", layerOutput.get_shape())
          return layerOutput
      def get_config(self):
          config = {'newShape': self.newShape,'newScale': self.newScale,'usemethod': self.usemethod}
          base_config = super(layerResize2D, self).get_config()
          return dict(list(base_config.items()) + list(config.items()))

class BCPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(BCPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_height,padding_width = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')
    # or 'REFLECT'
    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(BCPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#return global 
custom_objects={'BCPadding2D': BCPadding2D, 'layerResize2D': layerResize2D} 


def block_conv(input_conv, filters, alpha=3, beta=2, mode='ResNet',apply_BC=False):
      if len(input_conv.shape) == 5:
            if apply_BC:
                  from CustomLayers import BCPadding3D
                  padding = calc_pad_shape3D(input_conv.shape,input_conv.shape,alpha,beta)
                  #print("BCs with shape - padding ", padding)
                  conv = BCPadding3D(padding)(input_conv)
                  conv = layers.Conv3D(filters, kernel_size=alpha, strides=beta, padding='valid')(conv)
            else:
                conv = layers.Conv3D(filters, kernel_size=alpha, strides=beta, padding='same')(input_conv)
      elif len(input_conv.shape) == 4:
            if apply_BC:
                  from CustomLayers import BCPadding2D
                  padding = calc_pad_shape(input_conv.shape,input_conv.shape,alpha,beta)
                  #print("BCs with shape - padding ", padding)
                  conv = BCPadding2D(padding)(input_conv)
                  conv = layers.Conv2D(filters, kernel_size=alpha, strides=beta, padding='valid')(conv)
            else:
                  conv = layers.Conv2D(filters, kernel_size=alpha, strides=beta, padding='same')(input_conv)

      else:
            print(input_conv.shape, " appears to be an invalid layer for block_conv")
            #conv = input_conv
      if mode == 'GAN':
        conv = layers.BatchNormalization()(conv)
       
      return conv;


def block_deconv(conv, filters, method='Upsampling',apply_BC=False,alpha=3):
    if method == 'ConvT' :
        beta=2
        if len(conv.shape) == 5:
            conv = layers.Conv3DTranspose(filters, kernel_size=alpha, strides=beta, padding='same')(conv)
        elif len(conv.shape) == 4:
            conv = layers.Conv2DTranspose(filters, kernel_size=alpha, strides=beta, padding='same')(conv)
        else:
            print(conv.shape, " appears to be an invalid layer for layers.ConvNDTranspose")
    elif method == 'Upsampling' :
        if len(conv.shape) == 5:
            conv = layers.UpSampling3D(size=2, interpolation='bilinear')(conv) 
        elif len(conv.shape) == 4:
            conv = layers.UpSampling2D(size=2, interpolation='bilinear')(conv)
        else:
            print(conv.shape, " appears to be an invalid layer for layers.UpSamplingND")
        beta=1
        #conv = layers.Conv3D(filters, kernel_size=alpha, strides=beta, padding='same')(conv)
        conv = block_conv(conv, filters, alpha, beta, mode='NO_BN',apply_BC=apply_BC)
    

    elif method == 'Resize' :
        if len(conv.shape) == 5:
            from CustomLayers import layerResize3D
            conv = layerResize3D(newShape=None, newScale=2, usemethod='bilinear')(conv) 
        else:
            from CustomLayers import layerResize2D
            conv = layerResize2D(newShape=None, newScale=2, usemethod='bilinear')(conv) 
        beta=1
        conv = block_conv(conv, filters, alpha, beta, mode='NO_BN',apply_BC=apply_BC)

    else:
        print("Unknown Deconvolution Method \n Try: Resize2D , Upsampling, ConvT ")   
    return conv;


def activationLayer(inputs, activ_type='Prelu'):
    if activ_type == 'Prelu':
        outputs = layers.PReLU(alpha_initializer=initializers.Constant(value=0.25))(inputs)
    elif activ_type == 'Lrelu':
        outputs = layers.LeakyReLU(alpha=0.3)(inputs)
    else:
        #print("For activationLayer, INVALID activ_type=",activ_type, "\nSelecting layers.ReLU as activation" )
        outputs = layers.ReLU(max_value=None, negative_slope=0, threshold=0)(inputs)
    return outputs




def Callback_EarlyStopping(montiorLossValArr, min_delta=0.1, patience=10,verbose=False):
    """ Similar to tf.keras.callbacks.EarlyStopping
    #checks every epoch after 2*patience epoch 
    This works fine as long as the moving mean values are changing for patience epochs !
    patience is the number of epochs with no mean improvement after which break signal would be issued.
    Returns True or False signal. Break training loop if True  
    FUTURE: Implement cool down
    More details on: https://stackoverflow.com/a/63515365/7508007
    """
    recentEpoch, recentLossVal = len(montiorLossValArr) , montiorLossValArr[-1]
    
    if patience == 0:
        patience = 1
    #No early stopping for 2*patience epochs 
    if recentEpoch//patience < 2 :
        return False 
        
    #when lossVal stops decreasing  
    meanLoss_previousPatienceEpochs = np.mean(montiorLossValArr[::-1][patience:2*patience]) #for second-last patience epochs 
    meanLoss_recentPatienceEpochs = np.mean(montiorLossValArr[::-1][:patience]) #for last patience epochs
    delta_abs = np.abs(meanLoss_recentPatienceEpochs - meanLoss_previousPatienceEpochs )
    #added on 21 Aug at 02:17
    delta_abs =  np.abs(delta_abs / meanLoss_previousPatienceEpochs)  #percentage or relative change
    
    if verbose:
        print("Log Callback_EarlyStopping: Epoch & LossVal", recentEpoch, recentLossVal )
        print("Recent ",patience," epochs :",montiorLossValArr[::-1][:patience][::-1], " Mean: ", meanLoss_recentPatienceEpochs)
        print("Previous ",patience," epochs :",montiorLossValArr[::-1][patience:2*patience][::-1] , " Mean: ", meanLoss_previousPatienceEpochs)
        print("Change in loss value:", delta_abs, " Expected change:",min_delta)
        
    if delta_abs < min_delta :
        print("*Callback_EarlyStopping* Loss value didn't decrease from last ",patience," epochs.")
        print("*Callback_EarlyStopping* Change in loss value:", delta_abs, " Expected change:",min_delta)
        return True
    else:
        return False
#test montiorLossValArr = [1.0,1.001,1.00,1.22,1.22,1.22,1.22,1.22,1.22,1.22,1.22,1.22,1.25,1.22,1.26]
#  Callback_EarlyStopping(montiorLossValArr[:12], min_delta=1e-1, patience=5,verbose=True)
    


