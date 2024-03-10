# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:27:46 2023

@author: pcastror

This code fits predicted parameters from LS HCP to simulate images.

"""

### Load libraries
import numpy as np
from dipy.io.image import load_nifti
import time
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from utilities import vec, unvec, Zeppelin, bval_bvec_from_b_Matrix
import pickle as pk



idx_image = '01006'

path_GNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_WITH_correction_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1' + idx_image
path_noGNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_NO_correction_params.nii'
num = 72

################## no gnl correction #########################

with open('{}_mask.bin'.format(path_true_mask), 'rb') as file:
    msk= pk.load(file)
    
mask = msk[:, :, num]
print('mask shape: ', msk.shape)

loss_tissuetest, affine = load_nifti(path_noGNL_params)

loss_tissuetest2, affine = load_nifti(path_GNL_params)

file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\1'+ idx_image + '_DWI.txt'
b_matrix = np.loadtxt(file_path_bvec)
bval, g = bval_bvec_from_b_Matrix (b_matrix) 
bval=bval/1000

bval = bval[:108]
g = g[:108,:]
    

params = loss_tissuetest[:,:,num,:]

x_size = params.shape[0]
y_size = params.shape[1]

simulated_dwi_noGNL_3 = np.zeros((x_size,y_size,bval.shape[0]))

start_time = time.time()
for x in range(0,x_size,1):
        print(x)
        for y in range(0,y_size,1):
                
                signal = Zeppelin(params[x,y,:], g, bval)
                signal = signal.reshape(-1)
                
                simulated_dwi_noGNL_3[x,y,:] = signal
                
end_time = time.time()

#plt.imshow(simulated_dwi_noGNL[:,:,0])


################## with gnl correction #########################

path_grad_dev = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\1'+ idx_image + '_grad_dev.nii.gz'
grad_dev, affine = load_nifti(path_grad_dev)
print('grad_dev shape:', grad_dev.shape)

grad_dev = vec(grad_dev,msk)
grad_dev = grad_dev[0]
grad_dev = grad_dev.T

print(grad_dev.shape)

Lxx = grad_dev[:, 0]
Lxy = grad_dev[:, 1]
Lxz = grad_dev[:, 2]
Lyy = grad_dev[:, 3]
Lyx = grad_dev[:, 4]
Lyz = grad_dev[:, 5]
Lzz = grad_dev[:, 6]
Lzx = grad_dev[:, 7]
Lzy = grad_dev[:, 8] 


params = loss_tissuetest2


path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1'+idx_image+'_mask.bin'
with open(path_msk, 'rb') as file:
        msk = pk.load(file) 
		
params = vec(params,msk)
params = params[0]
params = params.T

print(params.shape)


pixels = params.shape[0] 

simulated_dwi_GNL_3 = np.zeros((pixels,bval.shape[0]))



for pixel in range(0,pixels,1):
            # construct L matrix
            L = np.array([[Lxx[pixel], Lxy[pixel], Lxz[pixel]],
                              [Lyx[pixel], Lyy[pixel], Lyz[pixel]],
                              [Lzx[pixel], Lxy[pixel], Lzz[pixel]]])
             
            # Apply gradient non-linearities correction
            I = np.eye(3)                          # identity matrix               
            v = np.dot(g, (I + L))
            n = np.sqrt(np.diag(np.dot(v, v.T)))
            
            new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
            new_bval = n ** 2 * bval
            new_bvec[new_bval == 0, :] = 0
            bval_corrected = new_bval 
            g_effective = new_bvec
			
            signal = Zeppelin(params[pixel,:], g_effective, bval_corrected)
            signal = signal.reshape(-1)
               
            simulated_dwi_GNL_3[pixel,:] = signal


simulated_dwi_GNL_3_ = unvec(simulated_dwi_GNL_3.T,msk)
print(simulated_dwi_GNL_3_.shape)

#plt.imshow(simulated_dwi_GNL_3_[:,:,num,0])

diff_w = 107
plt.imshow(np.rot90(simulated_dwi_noGNL_3[:,:,diff_w] - simulated_dwi_GNL_3_[:,:,num,diff_w]), cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.show()



idx_image = '00307'

path_GNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_WITH_correction_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1' + idx_image
path_noGNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_NO_correction_params.nii'
num = 72

################## no gnl correction #########################

with open('{}_mask.bin'.format(path_true_mask), 'rb') as file:
    msk= pk.load(file)
    
mask = msk[:, :, num]
print('mask shape: ', msk.shape)

loss_tissuetest, affine = load_nifti(path_noGNL_params)

loss_tissuetest2, affine = load_nifti(path_GNL_params)

file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\1'+ idx_image + '_DWI.txt'
b_matrix = np.loadtxt(file_path_bvec)
bval, g = bval_bvec_from_b_Matrix (b_matrix) 
bval=bval/1000

bval = bval[:108]
g = g[:108,:]
    

params = loss_tissuetest[:,:,num,:]

x_size = params.shape[0]
y_size = params.shape[1]

simulated_dwi_noGNL_1 = np.zeros((x_size,y_size,bval.shape[0]))

start_time = time.time()
for x in range(0,x_size,1):
        print(x)
        for y in range(0,y_size,1):
                
                signal = Zeppelin(params[x,y,:], g, bval)
                signal = signal.reshape(-1)
                
                simulated_dwi_noGNL_1[x,y,:] = signal
                
end_time = time.time()

#plt.imshow(simulated_dwi_noGNL[:,:,0])


################## with gnl correction #########################

path_grad_dev = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\1'+ idx_image + '_grad_dev.nii.gz'
grad_dev, affine = load_nifti(path_grad_dev)
print('grad_dev shape:', grad_dev.shape)

grad_dev = vec(grad_dev,msk)
grad_dev = grad_dev[0]
grad_dev = grad_dev.T

print(grad_dev.shape)

Lxx = grad_dev[:, 0]
Lxy = grad_dev[:, 1]
Lxz = grad_dev[:, 2]
Lyy = grad_dev[:, 3]
Lyx = grad_dev[:, 4]
Lyz = grad_dev[:, 5]
Lzz = grad_dev[:, 6]
Lzx = grad_dev[:, 7]
Lzy = grad_dev[:, 8] 


params = loss_tissuetest2


path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1'+idx_image+'_mask.bin'
with open(path_msk, 'rb') as file:
        msk = pk.load(file) 
		
params = vec(params,msk)
params = params[0]
params = params.T

print(params.shape)


pixels = params.shape[0] 

simulated_dwi_GNL_1 = np.zeros((pixels,bval.shape[0]))



for pixel in range(0,pixels,1):
            # construct L matrix
            L = np.array([[Lxx[pixel], Lxy[pixel], Lxz[pixel]],
                              [Lyx[pixel], Lyy[pixel], Lyz[pixel]],
                              [Lzx[pixel], Lxy[pixel], Lzz[pixel]]])
             
            # Apply gradient non-linearities correction
            I = np.eye(3)                          # identity matrix               
            v = np.dot(g, (I + L))
            n = np.sqrt(np.diag(np.dot(v, v.T)))
            
            new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
            new_bval = n ** 2 * bval
            new_bvec[new_bval == 0, :] = 0
            bval_corrected = new_bval 
            g_effective = new_bvec
			
            signal = Zeppelin(params[pixel,:], g_effective, bval_corrected)
            signal = signal.reshape(-1)
               
            simulated_dwi_GNL_1[pixel,:] = signal


simulated_dwi_GNL_1_ = unvec(simulated_dwi_GNL_1.T,msk)
print(simulated_dwi_GNL_1_.shape)

#plt.imshow(simulated_dwi_GNL_1_[:,:,num,0])

diff_w = 107
plt.imshow(np.rot90(simulated_dwi_noGNL_1[:,:,diff_w] - simulated_dwi_GNL_1_[:,:,num,diff_w]), cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.show()







idx_image = '00408'

path_GNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_WITH_correction_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1' + idx_image
path_noGNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_NO_correction_params.nii'
num = 72

################## no gnl correction #########################

with open('{}_mask.bin'.format(path_true_mask), 'rb') as file:
    msk= pk.load(file)
    
mask = msk[:, :, num]
print('mask shape: ', msk.shape)

loss_tissuetest, affine = load_nifti(path_noGNL_params)

loss_tissuetest2, affine = load_nifti(path_GNL_params)

file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\1'+ idx_image + '_DWI.txt'
b_matrix = np.loadtxt(file_path_bvec)
bval, g = bval_bvec_from_b_Matrix (b_matrix) 
bval=bval/1000

bval = bval[:108]
g = g[:108,:]
    

params = loss_tissuetest[:,:,num,:]

x_size = params.shape[0]
y_size = params.shape[1]

simulated_dwi_noGNL_2 = np.zeros((x_size,y_size,bval.shape[0]))

start_time = time.time()
for x in range(0,x_size,1):
        print(x)
        for y in range(0,y_size,1):
                
                signal = Zeppelin(params[x,y,:], g, bval)
                signal = signal.reshape(-1)
                
                simulated_dwi_noGNL_2[x,y,:] = signal
                
end_time = time.time()

#plt.imshow(simulated_dwi_noGNL[:,:,0])


################## with gnl correction #########################

path_grad_dev = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\1'+ idx_image + '_grad_dev.nii.gz'
grad_dev, affine = load_nifti(path_grad_dev)
print('grad_dev shape:', grad_dev.shape)

grad_dev = vec(grad_dev,msk)
grad_dev = grad_dev[0]
grad_dev = grad_dev.T

print(grad_dev.shape)

Lxx = grad_dev[:, 0]
Lxy = grad_dev[:, 1]
Lxz = grad_dev[:, 2]
Lyy = grad_dev[:, 3]
Lyx = grad_dev[:, 4]
Lyz = grad_dev[:, 5]
Lzz = grad_dev[:, 6]
Lzx = grad_dev[:, 7]
Lzy = grad_dev[:, 8] 


params = loss_tissuetest2


path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1'+idx_image+'_mask.bin'
with open(path_msk, 'rb') as file:
        msk = pk.load(file) 
		
params = vec(params,msk)
params = params[0]
params = params.T

print(params.shape)


pixels = params.shape[0] 

simulated_dwi_GNL_2 = np.zeros((pixels,bval.shape[0]))



for pixel in range(0,pixels,1):
            # construct L matrix
            L = np.array([[Lxx[pixel], Lxy[pixel], Lxz[pixel]],
                              [Lyx[pixel], Lyy[pixel], Lyz[pixel]],
                              [Lzx[pixel], Lxy[pixel], Lzz[pixel]]])
             
            # Apply gradient non-linearities correction
            I = np.eye(3)                          # identity matrix               
            v = np.dot(g, (I + L))
            n = np.sqrt(np.diag(np.dot(v, v.T)))
            
            new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
            new_bval = n ** 2 * bval
            new_bvec[new_bval == 0, :] = 0
            bval_corrected = new_bval 
            g_effective = new_bvec
			
            signal = Zeppelin(params[pixel,:], g_effective, bval_corrected)
            signal = signal.reshape(-1)
               
            simulated_dwi_GNL_2[pixel,:] = signal


simulated_dwi_GNL_2_ = unvec(simulated_dwi_GNL_2.T,msk)
print(simulated_dwi_GNL_2_.shape)

#plt.imshow(simulated_dwi_GNL_2_[:,:,num,0])

diff_w = 107
plt.imshow(np.rot90(simulated_dwi_noGNL_2[:,:,diff_w] - simulated_dwi_GNL_2_[:,:,num,diff_w]), cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.show()






idx_image = '01107'

path_GNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_WITH_correction_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1' + idx_image
path_noGNL_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_1' + idx_image + '_dwi_GNL_NO_correction_params.nii'
num = 72

################## no gnl correction #########################

with open('{}_mask.bin'.format(path_true_mask), 'rb') as file:
    msk= pk.load(file)
    
mask = msk[:, :, num]
print('mask shape: ', msk.shape)

loss_tissuetest, affine = load_nifti(path_noGNL_params)

loss_tissuetest2, affine = load_nifti(path_GNL_params)

file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\1'+ idx_image + '_DWI.txt'
b_matrix = np.loadtxt(file_path_bvec)
bval, g = bval_bvec_from_b_Matrix (b_matrix) 
bval=bval/1000

bval = bval[:108]
g = g[:108,:]
    

params = loss_tissuetest[:,:,num,:]

x_size = params.shape[0]
y_size = params.shape[1]

simulated_dwi_noGNL_4 = np.zeros((x_size,y_size,bval.shape[0]))

start_time = time.time()
for x in range(0,x_size,1):
        print(x)
        for y in range(0,y_size,1):
                
                signal = Zeppelin(params[x,y,:], g, bval)
                signal = signal.reshape(-1)
                
                simulated_dwi_noGNL_4[x,y,:] = signal
                
end_time = time.time()

#plt.imshow(simulated_dwi_noGNL[:,:,0])


################## with gnl correction #########################

path_grad_dev = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\1'+ idx_image + '_grad_dev.nii.gz'
grad_dev, affine = load_nifti(path_grad_dev)
print('grad_dev shape:', grad_dev.shape)

grad_dev = vec(grad_dev,msk)
grad_dev = grad_dev[0]
grad_dev = grad_dev.T

print(grad_dev.shape)

Lxx = grad_dev[:, 0]
Lxy = grad_dev[:, 1]
Lxz = grad_dev[:, 2]
Lyy = grad_dev[:, 3]
Lyx = grad_dev[:, 4]
Lyz = grad_dev[:, 5]
Lzz = grad_dev[:, 6]
Lzx = grad_dev[:, 7]
Lzy = grad_dev[:, 8] 


params = loss_tissuetest2


path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_1'+idx_image+'_mask.bin'
with open(path_msk, 'rb') as file:
        msk = pk.load(file) 
		
params = vec(params,msk)
params = params[0]
params = params.T

print(params.shape)


pixels = params.shape[0] 

simulated_dwi_GNL_4 = np.zeros((pixels,bval.shape[0]))



for pixel in range(0,pixels,1):
            # construct L matrix
            L = np.array([[Lxx[pixel], Lxy[pixel], Lxz[pixel]],
                              [Lyx[pixel], Lyy[pixel], Lyz[pixel]],
                              [Lzx[pixel], Lxy[pixel], Lzz[pixel]]])
             
            # Apply gradient non-linearities correction
            I = np.eye(3)                          # identity matrix               
            v = np.dot(g, (I + L))
            n = np.sqrt(np.diag(np.dot(v, v.T)))
            
            new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
            new_bval = n ** 2 * bval
            new_bvec[new_bval == 0, :] = 0
            bval_corrected = new_bval 
            g_effective = new_bvec
			
            signal = Zeppelin(params[pixel,:], g_effective, bval_corrected)
            signal = signal.reshape(-1)
               
            simulated_dwi_GNL_4[pixel,:] = signal


simulated_dwi_GNL_4_ = unvec(simulated_dwi_GNL_4.T,msk)
print(simulated_dwi_GNL_4_.shape)

#plt.imshow(simulated_dwi_GNL_4_[:,:,num,0])

diff_w = 107
plt.imshow(np.rot90(simulated_dwi_noGNL_4[:,:,diff_w] - simulated_dwi_GNL_4_[:,:,num,diff_w]), cmap = 'gray')
plt.axis('off')
plt.colorbar()
plt.show()





### print results



print('')

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
axes[0].imshow(np.rot90(simulated_dwi_noGNL_1[:,:,diff_w] - simulated_dwi_GNL_1_[:,:,num,diff_w]), cmap='gray', vmax = 250)
axes[0].set_title('Subject 1', fontsize = 25)
axes[0].axis('off')
axes[1].imshow(np.rot90(simulated_dwi_noGNL_2[:,:,diff_w] - simulated_dwi_GNL_2_[:,:,num,diff_w]), cmap='gray',vmax = 250 )
axes[1].set_title('Subject 2', fontsize = 25)
axes[1].axis('off')
axes[2].imshow(np.rot90(simulated_dwi_noGNL_3[:,:,diff_w] - simulated_dwi_GNL_3_[:,:,num,diff_w]), cmap='gray', vmax = 250)
axes[2].set_title('Subject 3', fontsize = 25)
axes[2].axis('off')
im = axes[3].imshow(np.rot90(simulated_dwi_noGNL_4[:,:,diff_w] - simulated_dwi_GNL_4_[:,:,num,diff_w]), cmap='gray', vmax = 250)
axes[3].set_title('Subject 4', fontsize = 25)
axes[3].axis('off')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(labelcolor='black', labelsize=18)
plt.tight_layout()
plt.show()




