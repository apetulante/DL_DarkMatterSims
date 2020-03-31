import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

init_func = 'he_normal'

# build the model layers
#input_size = (48,48,48,4)
#smallest_layer_size = (12,12,12,1)

def create_model(input_size):
  input_images = Input(input_size)

  conv1 = Conv3D(96, kernel_size=(3,3,3), strides = 1, activation = 'relu', padding = 'same', kernel_initializer = init_func)(input_images)
  conv1 = Conv3D(96, kernel_size=(3,3,3), strides = 1, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv1)
  conv1 = Conv3D(96, kernel_size=(3,3,3), strides = 1, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
  conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(pool1)
  conv2 = Conv3D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv2)
  #drop2 = Dropout(0.2)(conv2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
  conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(pool2)
  conv3 = Conv3D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv3)
  conv3 = Conv3D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv3)
  #drop3 = Dropout(0.2)(conv3)

  # construct input based on smallest layer size
  smallest_layer_size = conv3.get_shape().as_list()[1:4]
  smallest_layer_size.append(1)
  smallest_layer_shape = tuple(smallest_layer_size)
  input_labels = Input(smallest_layer_size)

  merge1 = concatenate([conv3,input_labels], axis = 4)

  up8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(UpSampling3D(size = (2,2,2))(merge1))
  merge8 = concatenate([conv2,up8], axis = 4)
  conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(merge8)
  conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv8)

  up9 = Conv3D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(UpSampling3D(size = (2,2,2))(conv8))
  merge9 = concatenate([conv1,up9], axis = 4)
  conv9 = Conv3D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(merge9)
  conv9 = Conv3D(96, 3, activation = 'relu', padding = 'same', kernel_initializer = init_func)(conv9)

  output = Conv3D(1,1, activation = 'relu', padding='same')(conv9)

# compile the model
  model = Model(inputs = [input_images,input_labels], outputs = output)
  return model

#adam_opt = Adam(lr=0.00001, beta_1=0.999, beta_2=0.9999, amsgrad=False)
#model.compile(optimizer = adam_opt, loss = 'mse', metrics = ['mae'])
#print(model.summary(line_length=120))

