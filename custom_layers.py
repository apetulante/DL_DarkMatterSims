import tensorlfow as tf
import numpy as np
from tensorflow.keras.layers import Layer


def outside_pad_3D(inputs, trunc_input_sz):
     current_sz = inputs[0].shape
     model_input_sz = inputs[1].shape
     pixel_sz = int(current_sz/truc_input_sz)

     # pad the image with zeros first
     zero_padded_im = tf.pad(inputs[0], mode='constant', value=0)

     #then, create a "shell" of the original image
     roi_high_bound = trunc_input_sz+pixel_sz   #higher bound of the region of interest of the image to cut out
     roi_low_bound = model_input_sz - trunc_input_sz - pixel_sz #lower bound of the region of interest of the image to cut out
     shell = inputs[1][:,roi_low_bound:roi_high_bound,roi_low_bound:roi_high_bound,roi_low_bound:roi_high_bound,:]

     shell[:,pixel_sz:-pixel_sz,pixel_sz:-pixel_sz,pixel_sz:-pixel_sz,:] = 0

     #pool it to the correct size
     pad_values = tf.pool(shell, size = pixel_sz)

     #add them together
     padded_im = zero_padded_im + pad_values



class Truncate(Layer):

class OutsidePadding3D(Layer):
   def __init__(self, padding=(1, 1), model_input_shape, **kwargs):
     self.padding = tuple(padding)
     self.padding_sz = 
     self.input_spec = [InputSpec(ndim=4)]
     super(ReflectionPadding2D, self).__init__(**kwargs)
  
   def call(inputs, trunc_input_sz):
     current_sz = inputs[0].shape
     model_input_sz = inputs[1].shape
     pixel_sz = current_sz/truc_input_sz
 
     # pad the image with zeros first
     zero_padded_im = tf.pad(inputs[0], mode='constant', value=0)

     #then, create a "shell" of the original image
     roi_high_bound = trunc_input_sz+pixel_sz   #higher bound of the region of interest of the image to cut out
     roi_low_bound = model_input_sz - trunc_input_sz - pixel_sz #lower bound of the region of interest of the image to cut out
     shell = inputs[1][:,roi_low_bound:roi_high_bound,roi_low_bound:roi_high_bound,roi_low_bound:roi_high_bound,:]
     
     shell[:,pixel_sz:-pixel_sz,pixel_sz:-pixel_sz,pixel_sz:-pixel_sz,:] = 0

     #pool it to the correct size
     pad_values = tf.pool(shell, size = pixel_sz)

     #add them together
     padded_im = zero_padded_im + pad_values


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
