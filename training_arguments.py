from loss_functions import *
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.optimizers import *

# DEFINE MODEL PARAMETERS
#model_path = '/scratch/petulaa/model_CURRENT/model2.h5'
#model_path = '/scratch/petulaa/model/current_model'
model_path = '/scratch/petulaa/model/model.h5' 
loss_history_filepath = '/scratch/petulaa/model/loss_history.txt'
custom_objects_dict = dict({'constraint_component': constraint_component})

# INFORMATION ABOUT THE DATA
dims = 4 #number of dimensions (in order: pos, vel x, vel y, vel z) you want to use
#input_size = (96,96,96,dims)  #size of the data to crop the INPUT to. Must be <= actual size of data, for obvious reasons.
			      # IF LOADING AN ALREADY EXISTING MODEL: this will be re-defined to whatever shape that model used.
output_size = (48,48,48,dims)  #size of the data to crop the OUTPUT to. Must be <= actual size of data, for obvious reasons.
  			      # **FOR MOST APPLICATIONS: this should be the same as input size**
                              # IF LOADING AN ALREADY EXISTING MODEL: this will be re-defined to whatever shape that model used.
input_size = output_size

n_examples = 100 # number of examples to use PER SNAPSHOT

input_paths = [#"/scratch/petulaa/snap_box_CarmenF_z0p186_100MpcBoxes"]
               "/scratch/petulaa/snap_box_CarmenF_z0p298_withVels",
               "/scratch/petulaa/snap_box_CarmenF_z0p499_withVels"]
output_path = "/scratch/petulaa/snap_box_CarmenF_z0p000_withVels"

# DEFINE TRAINING PARAMETERS
n_epochs = 2
loss_func = 'mae'
metrics_list = ['mse', constraint_component]

# FOR **NEW** MODELS, DEFINE HYPERS + OPTIMIZER
learning_rate = 0.00001
decay = 1
b1 = 0.999 #the first beta value for momentum optimizers
b2 = 0.9999 #the second beta value

optimizer = Adam(lr=learning_rate, beta_1=b1, beta_2=b2, decay = decay, amsgrad=False)

# DEFINE FILEPATHS TO SAVE OUTPUT
output_figures_filepath = '/scratch/petulaa/model/figures/'
output_model_filepath = '/scratch/petulaa/model'
