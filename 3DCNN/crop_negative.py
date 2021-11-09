from glob import glob
from nilearn import image
import scipy.io as sio 
import numpy as np
import random
import os

# caminho do banco de dados 
data_path = 'cmb-3dcnn-data'
data_files = [filee for filee in sorted(glob(data_path + '/nii/*.nii'))][:10]
print(data_files)

if not os.path.exists('3DCNN/data_crop/negative'): # criar a pasta caso não exista
    os.makedirs('3DCNN/data_crop/negative')

in_shape = [20,20,16] # tamanho do recorte
IN_X, IN_Y, IN_Z = [x//2 for x in in_shape]

list_roi = list()
f = 0
batch = 5  # quantidade de arquivos  .nii a serem processado em cada batch, se for muito grande pode da erro por causa da memoria
batchs_count = 0

for files in data_files:
    img = image.smooth_img(files, fwhm=3).get_data()
    for j in range(8):        
        x = random.randint(150,350)
        y = random.randint(150,350)
        z = random.randint(20,100)
        roi = img[x-IN_X:x+IN_X, y-IN_Y:y+IN_Y, z-IN_Z:z+IN_Z]

        list_roi.append(roi)
        rois = np.asarray(list_roi)

    f+=1

    if f == batchs_count + batch :        
        rois = np.asarray(list_roi)
        print(rois.shape)
        #np.save('3DCNN/data_crop/negative/negatives_%sx%sx%s.npy'%(in_shape[0],in_shape[1],in_shape[2]), rois) 
        np.save('3DCNN/data_crop/negative/negatives_%s-%s_%sx%sx%s.npy'%(batchs_count,batchs_count + batch,in_shape[0],in_shape[1],in_shape[2]), rois) 
        rois = None
        list_roi = []
        batchs_count += batch
        print('reset in batch: ', batchs_count)
            
# rois = np.asarray(list_roi)
# print(rois.shape)
# np.save('3DCNN/data_crop/negative/negatives_%sx%sx%s.npy'%(in_shape[0],in_shape[1],in_shape[2]), rois) 