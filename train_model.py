from load_model import *
from load_data import *
from loss_functions import *
from training_arguments import *
from generate_figs import *

import tensorflow as tf
from tensorflow.keras.models import *
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import sys
import argparse

# supress any tensorflow log messages that aren't errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

parser=argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of training epochs. Will default to whatever is in training_arguments.py file, but this argument will overwrite that value if used')
parser.add_argument('--model', help='Filepath to model to use. Will default to whatever is in training_arguments.py file, but this argument will overwrite that path if used. Use --model='' to use the model in load_model.py')
args=parser.parse_args()

if args.epochs != None:
  n_epochs = args.epochs
if args.model != None:
  model_path = args.model

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print(tf.__version__)
n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", n_gpus)
batch_size = 8*n_gpus

if model_path == '':
  if len(tf.config.experimental.list_physical_devices('GPU')):
  #tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      model = create_model(input_size)
      opt = optimizer#Adam(lr=0.0005, beta_1=0.999, beta_2=0.999, decay = 1, amsgrad=False)
      model.compile(optimizer = opt, loss = loss_func, metrics = metrics_list)#['mse','mae',constraint_component])
  loss_hist = []
  n_passed_epochs = 0
else:
  orig_model = load_model(model_path, custom_objects = custom_objects_dict)
  weights = orig_model.get_weights()
  opt_weights = orig_model.optimizer.get_weights()
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    cloned_model = tf.keras.models.clone_model(orig_model)
    cloned_model.compile(optimizer = orig_model.optimizer, loss = orig_model.loss, metrics = orig_model.metrics)
    cloned_model.set_weights(weights)
    cloned_model.optimizer.set_weights(opt_weights)
    model = cloned_model
  loss_hist = np.genfromtxt(loss_history_filepath)
  n_passed_epochs = len(loss_hist)

model.summary(line_length=120)

X, y, Xlabels, test_im_X, test_im_y, test_ims_Xlabels, all_filenums = load_data(model.input, model.output)

#if len(input_paths) > 1:

#for new_epochs in range(n_epochs):
#   print("Epoch %d" %(new_epochs+n_passed_epochs))
model_history =  model.fit([X, Xlabels], y, epochs = n_epochs, verbose = 1, batch_size = batch_size)
if len(loss_hist) == 0:
   loss_hist = model_history.history['loss']
else:
   loss_hist = np.append(loss_hist, model_history.history['loss'])
#print(loss_hist)
#else:   
#model.fit(X, y, epochs = n_epochs, verbose = 1, batch_size = batch_size)

data_num=1
model.predict([X, Xlabels])

if model_path == '':
   model.save('%s/model.h5' %output_model_filepath)
else:
   weights = model.get_weights()
   opt_weights = model.optimizer.get_weights()
   orig_model.set_weights(weights)
   orig_model.optimizer.set_weights(opt_weights)
   orig_model.save('%s/model.h5' %output_model_filepath)

np.savetxt(loss_history_filepath, loss_hist)

plot_loss_per_epoch(loss_hist)
plot_predicted_image(model, X, Xlabels, y, data_nums = [0,8])
plot_pixel_accuracy_heatmap(model, X, Xlabels, y)
