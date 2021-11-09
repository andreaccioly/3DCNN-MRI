import mat73
import numpy as np
import scipy.io as sio 
from glob import glob
from time import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D

path_3DCNN = '3DCNN/tf_model/mymodel63seq05_11' 
path_test_set_cand = '/home/offsouza/demo/result/test_set_cand/'
path_save_predictions = '3DCNN/final_predictions/'
print('CNN LOADING ...')       
CNN_model = tf.keras.models.load_model(path_3DCNN)
print('CNN LOADED')

start_time = time()
for num_case,case in enumerate(glob(path_test_set_cand + "/*.mat")):
    data = mat73.loadmat(case)['test_set_x']    
    print("predicting  {}_test_set_cand, contains {} candidates".format(num_case+1,data.shape[0]))       
    data_reshape = data.reshape(data.shape[0],20,20,16,1)
    predictions_prob = CNN_model.predict(data_reshape)
    predictions = np.asarray([[np.argmax(x) for x in predictions_prob]])
    sio.savemat(path_save_predictions + '{}_prediction.mat'.format(num_case+1), {'prediction':predictions})
print('time spent {} seconds.'.format(time()-start_time))