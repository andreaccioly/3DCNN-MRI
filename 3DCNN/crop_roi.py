from glob import glob
from nilearn import image
import scipy.io as sio 
import numpy as np
import os

# caminho do banco de dados 
data_path = 'cmb-3dcnn-data'


data_files = [filee for filee in sorted(glob(data_path + '/nii/*.nii'))][:]
print(data_files)

if not os.path.exists('3DCNN/data_crop/positive'): # criar a pasta caso não exista
    os.makedirs('3DCNN/data_crop/positive')

in_shape = [16,16,10] # tamanho do recorte
IN_X, IN_Y, IN_Z = [x//2 for x in in_shape]

g_truths = [filee for filee in sorted(glob(data_path + '/ground_truth/*.mat'))][:]

list_roi = list()
f = 0
batch = 5  # quantidade de arquivos  .nii a serem processado em cada batch, se for muito grande pode da erro por causa da memoria
batchs_count = 0

for gt, files in zip(g_truths, data_files):
    img = image.smooth_img(files, fwhm=3).get_data()
    array = sio.loadmat(gt)['cen']
    print(gt,files)
    for cen in array:
        x,y,z = cen
        
        roi = img[x-IN_X:x+IN_X, y-IN_Y:y+IN_Y, z-IN_Z:z+IN_Z]
        print(x,y,z, roi.shape)
        
        list_roi.append(roi)
        rois = np.asarray(list_roi)
        print(rois.shape)  
    f+=1        

    
    if f == batchs_count + batch :        
        rois = np.asarray(list_roi)
        print(rois.shape)
        np.save('3DCNN/data_crop/positive/roi_%s-%s_%sx%sx%s.npy'%(batchs_count,batchs_count + batch,in_shape[0],in_shape[1],in_shape[2]), rois) 
        rois = None
        list_roi = []
        batchs_count += batch
        print('reset in batch: ', batchs_count)
        
    # elif f == 20:
    #     np.save('3DCNN/roi_10-20_%sx%sx%s.npy'%(in_shape[0],in_shape[1],in_shape[2]), rois) 
    #     rois = None
    #     list_roi = []