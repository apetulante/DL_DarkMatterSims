from training_arguments import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

graph = tf.Graph()

def plot_pixel_accuracy_heatmap(model, X, Xlabels, y):
   print("Plotting the pixel heatmap")
   fig = plt.figure() 
   buffer_zone = 8
  
   im_shape = input_size[0]
   errs = np.zeros([im_shape,im_shape])
   slices = np.arange(buffer_zone,im_shape-buffer_zone,1)
   data_num = 0
   for prediction in model.predict([X, Xlabels]):
      if data_num>0:
         if data_num%20 == 0:
            print("Getting errors for image #%d" %data_num)
            np.savetxt('errs.txt',errs)
      for slice in slices:
         input_im = X[data_num,slice,:,:,0]
         output_im = y[data_num][slice].reshape(im_shape,im_shape)
         predicted_im = prediction[slice,:,:,0]
         errs += np.array(np.abs(predicted_im - output_im)/np.average(X))
         #errs += (predicted_im - output_im)**2
      data_num += 1
   plt.imshow(errs,cmap='Oranges')
   plt.savefig("%s/pixel_acc_heatmap.png" %output_figures_filepath)


def plot_loss_per_epoch(loss_hist):
   fig = plt.figure()
   plt.plot(np.arange(len(loss_hist)),loss_hist)
   plt.title("Training Loss / Epoch")
   plt.savefig("%s/loss_hist.png" %output_figures_filepath)


def plot_predicted_image(model, X, Xlabels, y, data_nums, scale="normal"): #optionally, scale = 'log'
   im_shape = input_size[0]

   im_num = 1
   for data_num in data_nums:
      X_redshift = np.average(Xlabels[data_num:data_num+1])

      fig = plt.figure(figsize=(15,5))
      ax1, ax2, ax3 = fig.subplots(1,3)

      if scale == 'log':
         input_im = np.log10(X[data_num,10,:,:,0]+1)
         output_im = np.log10(y[data_num][10].reshape(im_shape,im_shape)+1)
         predicted_im = np.log10(model.predict([X, Xlabels])[data_num:data_num+1][0,10,:,:,0]+1)
      else:
         input_im = X[data_num,10,:,:,0]
         output_im = y[data_num][10].reshape(im_shape,im_shape)
         print(X[data_num:data_num+1].shape)
         predicted_im = model.predict([X, Xlabels])[data_num:data_num+1][0,10,:,:,0]
   
      vmax = max(np.max(input_im),np.max(output_im),np.max(predicted_im))
      vmin = min(np.min(input_im),np.min(output_im),np.min(predicted_im))
   
      ax1.imshow(input_im, vmin=vmin, vmax=vmax)
      ax1.set_title("Input, z = %f" %(X_redshift/1000))
      ax2.imshow(output_im, vmin=vmin, vmax=vmax)
      ax2.set_title("Output Truth")
      ax3.imshow(predicted_im, vmin=vmin, vmax=vmax)
      ax3.set_title("Output Predicted")
   
      plt.savefig("%s/predicted_ims_%d_z%d.png" %(output_figures_filepath, im_num, X_redshift))
      im_num += 1

#def plot_accuracy_histogram():

