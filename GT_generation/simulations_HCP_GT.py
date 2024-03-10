# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:27:46 2023

@author: pcastror

This code fits GT parameters HCP to back-simulate images.

"""

### Load libraries
import numpy as np
from dipy.io.image import load_nifti
import time
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from utilities import Zeppelin, bval_bvec_from_b_Matrix


print('') 
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('Device used. using GPU (cuda) will significantly speed up the training:', device) #Device used: using GPU will significantly speed up the training.
   
 
# Read the image
file_path_params = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\PREDICTIONS_LS_params.nii'
file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\100307_DWI.txt'
params, affine = load_nifti(file_path_params)
b_matrix = np.loadtxt(file_path_bvec)
bval, g = bval_bvec_from_b_Matrix (b_matrix) 
bval=bval/1000

x_size = params.shape[0]
y_size = params.shape[1]
z_size = params.shape[2]

simulated_dwi = np.zeros((x_size,y_size,z_size,bval.shape[0]))

start_time = time.time()
for z in range(0,z_size,1):
    print('num slice ', z , '/', z_size)
    for x in range(0,x_size,1):
        for y in range(0,y_size,1):
            if not np.isnan(params[x, y, z, 0]):
                
                signal = Zeppelin(params[x,y,z,:], g, bval)
                signal = signal.reshape(-1)
                
                simulated_dwi[x,y,z,:] = signal
                
                
end_time = time.time()

print('')
print('Time to simulate signal from predicted parameters: ', start_time - end_time, 'sec')
print('')

# Read the image
file_path_nii = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\100307_DWI.nii'
real_dwi, affine = load_nifti(file_path_nii)   

num = 70
num_bval = 0

plt.imshow(simulated_dwi[:,:,num,num_bval], cmap = 'gray')
plt.title(f'Simulated signal from pred LS params. Slice {num}, bval {num_bval}')
plt.axis('off')
plt.colorbar()
plt.show()

plt.imshow(real_dwi[:,:,num,num_bval], cmap = 'gray')
plt.title(f'REAL signal from pred LS params. Slice {num}, bval {num_bval}')
plt.axis('off')
plt.colorbar()
plt.show()

plt.imshow(np.abs(real_dwi[:,:,num,num_bval] - simulated_dwi[:,:,num,num_bval]), cmap = 'gray')
plt.title(f'REAL -  simualted. Slice {num}, bval {num_bval}')
plt.axis('off')
plt.colorbar()
plt.show()

mask_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\PREDICTIONS_LS_mask.bin' 
import pickle as pk
with open(mask_file_name, 'rb') as file:
    mask = pk.load(file)


signals_diff = real_dwi[:,:,num,num_bval] - simulated_dwi[:,:,num,num_bval]
signals_diff = signals_diff[mask[:,:,num] == True]

plt.hist(signals_diff)
plt.title(f'REAL -  simualted. Slice {num}, bval {num_bval}')
plt.show()


output_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\SIMULATIONS_FROM_LS'

## save predicted params as a nii flie
file_path = output_file_name + '_dwi.nii'
nii_img = nib.Nifti1Image(simulated_dwi, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_img, file_path)

print('nii params saved!')
print('')









