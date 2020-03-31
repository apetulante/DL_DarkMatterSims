import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import time
import os

from load_model import *
from training_arguments import *

#input_paths = ["/scratch/petulaa/snap_box_CarmenF_z0p186_withVels",
#               "/scratch/petulaa/snap_box_CarmenF_z0p298_withVels",
#               "/scratch/petulaa/snap_box_CarmenF_z0p499_withVels"]
#output_path = "/scratch/petulaa/snap_box_CarmenF_z0p000_withVels"
#n_boxes = len(os.listdir(input_path))

in_sz = input_size[0]
out_sz = output_size[0]

def load_data(input_layers, output_layer):
   # start by getting the shape of the labels, (if multi-input model). Else, smallest layer shape will remain arbitrary (and unused)
   smallest_layer_shape = (1)
   for layer in input_layers:
      smallest_layer_shape = tuple(layer[1].shape.as_list())      
      print(smallest_layer_shape)
   #Define the lists we'll fill in with the data
   X = []
   y = []
   Xlabels = []

   all_filenums = []
   for input_path in input_paths:
       # first, get the numbered names of the files to be sure there's matches across input/output snaps
       # Get file numbers that we have for the input
       filenums_input = []
       filenames_input = os.listdir(input_path)
       for filename in filenames_input:
         if 'dim3' in filename:
           filenums_input.append(int(filename.split('_')[2].split('.')[0]))
       # Get file numbers we have for the putput
       filenums_output = []
       filenames_output = os.listdir(output_path)
       for filename in filenames_output:
         if 'dim3' in filename:
           filenums_output.append(int(filename.split('_')[2].split('.')[0]))
   
       filenums = np.intersect1d(np.array(filenums_input), np.array(filenums_output))
       np.random.shuffle(filenums) #randomize the files so we don't have too many repeats
   
       t0 = time.time()
   
       # Load data into the array for this snapshot
       for filenum in filenums[0:n_examples]:
         X_data = []
         y_data = []
         for dim_num in range(dims):
             X_data_raw = np.genfromtxt('%s/dim%d_box_%d.txt' %(input_path, dim_num, filenum), delimiter = ' ')
             y_data_raw = np.genfromtxt('%s/dim%d_box_%d.txt'%(output_path, dim_num, filenum), delimiter = ' ')
             X_sz = X_data_raw.shape[-1]  # get the actual size of the images, so we can reshape them into matricies
             y_sz = y_data_raw.shape[-1]
             X_data.append(X_data_raw.reshape(X_sz,X_sz,X_sz)[0:in_sz,0:in_sz,0:in_sz]) # crop to out_sz if wanted
             y_data.append(y_data_raw.reshape(y_sz,y_sz,y_sz)[0:out_sz,0:out_sz,0:out_sz])
   
         X.append(np.array(X_data).T)
         y.append(np.array(y_data).T)

         #make redshift label array
         redshift = int(input_path.split('_')[3].split('p')[1])
         Xlabels.append(np.ones(smallest_layer_shape)*redshift*10)  
          
         # print occasional time updates 
         if len(X)%20 == 0:
             t1 = time.time() - t0
             print("Time Elapsed: ",t1)
             print('%d files loaded' %len(X))
             t0 = time.time()
       all_filenums.append(filenums[0:n_examples])
   
   X = np.array(X)#.reshape(n_examples, out_sz, out_sz, out_sz, dims)
   y = np.array(y)#.reshape(n_examples, out_sz, out_sz, out_sz, dims)
   Xlabels = np.array(Xlabels)
   
   print('X: ', X.shape)
   print('y: ', y.shape)
   print('Xlabels: ', Xlabels.shape)
   

   # Get an image to use for testing/ display purposes
   test_im_num = np.unique(np.append(filenums,all_filenums))[20]

   # Set up test image arrays
   test_ims_X = []
   test_ims_y = []
   test_ims_Xlabels = []
   
   # get the same test image for all snapshots
   for input_path in input_paths:
     test_im = []
     for dim_num in range(dims):
       test_X_raw = np.genfromtxt('%s/dim%d_box_%d.txt' %(input_path, dim_num, test_im_num), delimiter = ' ')
       test_im.append(test_X_raw.reshape(X_sz,X_sz,X_sz)[0:in_sz,0:in_sz,0:in_sz])
     test_ims_X.append(np.array(test_im).T)  
     redshift = int(input_path.split('_')[3].split('p')[1])
     test_ims_Xlabels.append(np.ones(smallest_layer_shape)*redshift*10)
   
   test_y_raw = np.genfromtxt('%s/dim%d_box_%d.txt' %(output_path, 0, test_im_num), delimiter = ' ')
   test_im_y = test_y_raw.reshape(y_sz,y_sz,y_sz)[0:out_sz,0:out_sz,0:out_sz].T
   
   test_ims_X = np.array(test_ims_X)
   test_im_y = np.array(test_im_y)
   test_ims_Xlabels = np.array(test_ims_Xlabels)
   
   print('X test: ', test_ims_X.shape)
   print('y test: ', test_im_y.shape)
   print('X test labels: ', test_ims_Xlabels.shape)
   
   # Delete extra y dimensions from the output if not predicting all dimensions
   out_dims =  output_layer.shape.as_list()[-1]
   
   y_new = []
   for im_num in range(len(y)):
     y_new.append(y[im_num,:,:,:,0:out_dims])
     
   y = np.array(y_new)
   print("new y shape:", y.shape) 
   #test_im_y = test_im_y.T[0].reshape(1,out_sz,out_sz,out_sz,1)
   #print(test_im_y.shape)
   
   return X, y, Xlabels, test_ims_X, test_im_y, test_ims_Xlabels, all_filenums
