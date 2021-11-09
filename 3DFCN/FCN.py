import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU

import tensorflow as tf
import numpy as np
from glob import glob
from nilearn import image
import scipy.io as sio 

from wrapper import FCN_wrapper #cnn2fcn, clip_volume_data

# Parametros

path_3DCNN = '3DCNN/tf_model/mymodel78seq06_09'  # caminho do modelo 3DCNN treinado
data_path = 'cmb-3dcnn-data/nii' # caminho da pasta com os arquivos .nii
qnt_data = 1 # quantidade de arquivos a serem lidos. Para ler todos os arquivos colocar -1
save_path_score_mask = '3DFCN/score_mask_FCN/' #arquivo onde será salvo os resultado score_mask

if not os.path.exists(save_path_score_mask): # criar a pasta caso não exista
    os.makedirs(save_path_score_mask)
fcn = FCN_wrapper(path_3DCNN)
# Convert 3DCNN to 3DFCN
fcn.cnn2fcn()
files = 0 
for filee in sorted(glob(data_path + '/*.nii')):
    img = image.smooth_img(filee, fwhm=3).get_data() 
    img = img.reshape(1,512,512,150,1)                     
    score = fcn.clip_and_predict(img)
    np.save(save_path_score_mask+ 'score_mask_%s.npy'%files,score)
    data = np.load(save_path_score_mask+ 'score_mask_%s.npy'%files)
    sio.savemat(save_path_score_mask +'%s_score_mask.mat'%files,{'score_mask':data.reshape(2,249,249,70)})
    print(score.shape)
    files+=1
    if files == qnt_data:
        break

