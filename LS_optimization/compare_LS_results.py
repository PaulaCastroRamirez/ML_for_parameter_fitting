# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:19:59 2023

@author: pcastror

This script evaluates LS fitting with vs without grad_dev correction
"""

import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from dipy.reconst.dti import fractional_anisotropy
from dipy.io.image import load_nifti

mask_path = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\estimated_params_no_correction_try1_ALL_100307_DWI_mask.bin'
path_niter_true_nc = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\niter5_correct2_params.nii'
path_niter_true_c = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\niter5_corrected2_grad_params.nii'

num = 70 

true_test_param2, affine = load_nifti(path_niter_true_nc) 
true_test_params_unvec_nc = true_test_param2[:,:,num,:]
print('true_test_params_unvec_nc shape: ', true_test_params_unvec_nc.shape)

true_test_param2, affine = load_nifti(path_niter_true_c) 
true_test_params_unvec_c = true_test_param2[:,:,num,:]
print('true_test_params_unvec_c shape: ', true_test_params_unvec_c.shape)

with open(mask_path, 'rb') as file:
    msk = pk.load(file)
    
mask_test = msk[:,:,num]


#################### EVALUATE REAL IMAGE PREDICTIONS VS ML PREDICTIONS #########################
############### GT LS IMAGES ####################
axial_diffusivity= true_test_params_unvec_nc[:,:,2]
radial_diffusivity = true_test_params_unvec_nc[:,:,3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_iter_nc = fractional_anisotropy(diffusion_tensor) * mask_test
MD_iter_nc =  ((axial_diffusivity + radial_diffusivity + radial_diffusivity) / 3) * mask_test

## predicted angles
# Define the size of your image
height = true_test_params_unvec_nc.shape[0]
width = true_test_params_unvec_nc.shape[1]

theta = true_test_params_unvec_nc[:,:,0]
phi = true_test_params_unvec_nc[:,:,1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_true = np.zeros((height,width, 3), dtype=np.uint8)
rgb_map_true[:, :, 0] = x_abs * 255   # Red channel
rgb_map_true[:, :, 1] = y_abs * 255   # Green channel
rgb_map_true[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_true[:,:,0] = np.where(mask_test == 1, rgb_map_true[:,:,0], np.nan)
rgb_map_true[:,:,1] = np.where(mask_test == 1, rgb_map_true[:,:,1], np.nan)
rgb_map_true[:,:,2] = np.where(mask_test == 1, rgb_map_true[:,:,2], np.nan)

rgb_map_true2_nc = np.zeros((height,width, 3), dtype=np.uint8)
rgb_map_true2_nc[:, :, 0] = rgb_map_true[:, :, 0] * FA_iter_nc   # Red channel
rgb_map_true2_nc[:, :, 1] = rgb_map_true[:, :, 1] * FA_iter_nc   # Green channel
rgb_map_true2_nc[:, :, 2] = rgb_map_true[:, :, 2] * FA_iter_nc   # Blue channel

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle('Results from fitting with LS python code no correction', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(true_test_params_unvec_nc[:,:,4]* mask_test), cmap='gray')
plt.title('S0', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(true_test_params_unvec_nc[:,:,2]* mask_test), cmap='gray')
plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(true_test_params_unvec_nc[:,:,3]* mask_test), cmap='gray')
plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_iter_nc), cmap='gray')
plt.title('FA map', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_iter_nc), cmap='gray')
plt.title('MD [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_true2_nc), cmap='jet')
# RGB-encoded principal direction
plt.title('rgb map principal direction º', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.show()


# =============================================================================
# with open('{}\HCP_small_sigtest.bin'.format(path_order_true), 'rb') as file:
#     # Load the object from the 
#     file_truetest = os.path.basename(file.name)
#     true_test_signal = pk.load(file)
# =============================================================================
    

#################### EVALUATE LS + grad_dev predictions #########################

axial_diffusivity= true_test_params_unvec_c[:,:,2]
radial_diffusivity = true_test_params_unvec_c[:,:,3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_iter_c = fractional_anisotropy(diffusion_tensor) * mask_test
MD_iter_c =  ((axial_diffusivity + radial_diffusivity + radial_diffusivity) / 3) * mask_test

## predicted angles
# Define the size of your image
height = true_test_params_unvec_c.shape[0]
width = true_test_params_unvec_c.shape[1]

theta = true_test_params_unvec_c[:,:,0]
phi = true_test_params_unvec_c[:,:,1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_true = np.zeros((height,width, 3), dtype=np.uint8)
rgb_map_true[:, :, 0] = x_abs * 255   # Red channel
rgb_map_true[:, :, 1] = y_abs * 255   # Green channel
rgb_map_true[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_true[:,:,0] = np.where(mask_test == 1, rgb_map_true[:,:,0], np.nan)
rgb_map_true[:,:,1] = np.where(mask_test == 1, rgb_map_true[:,:,1], np.nan)
rgb_map_true[:,:,2] = np.where(mask_test == 1, rgb_map_true[:,:,2], np.nan)

rgb_map_true2_c = np.zeros((height,width, 3), dtype=np.uint8)
rgb_map_true2_c[:, :, 0] = rgb_map_true[:, :, 0] * FA_iter_c   # Red channel
rgb_map_true2_c[:, :, 1] = rgb_map_true[:, :, 1] * FA_iter_c   # Green channel
rgb_map_true2_c[:, :, 2] = rgb_map_true[:, :, 2] * FA_iter_c   # Blue channel

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle('Results from fitting with LS python code with correction', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(true_test_params_unvec_c[:,:,4]* mask_test), cmap='gray')
plt.title('S0', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(true_test_params_unvec_c[:,:,2]* mask_test), cmap='gray')
plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(true_test_params_unvec_c[:,:,3]* mask_test), cmap='gray')
plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_iter_c), cmap='gray')
plt.title('FA map', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_iter_c), cmap='gray')
plt.title('MD [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_true2_c), cmap='jet')
plt.title('rgb_map principal direction º', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.show()

rgb_map_diff = np.empty((rgb_map_true2_c.shape[0], rgb_map_true2_c.shape[1]))
rgb_map_diff3 = np.empty((rgb_map_true2_nc.shape[0], rgb_map_true2_nc.shape[1]))

dot_product = np.empty((rgb_map_true2_c.shape[0], rgb_map_true2_c.shape[1]))
for pix_x in range(0,rgb_map_true2_c.shape[0],1):
    for pix_y in range(0,rgb_map_true2_c.shape[1],1):
        vector1 = np.array(rgb_map_true2_c[pix_x,pix_y,:])
        vector1 = np.double(vector1)
        vector2 = np.array(rgb_map_true2_nc[pix_x,pix_y,:])
        vector2 = np.double(vector2)
        dot_product[pix_x,pix_y] = np.dot(vector1, vector2)

        rgb_map_diff[pix_x,pix_y] = np.arccos(dot_product[pix_x,pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
        rgb_map_diff3[pix_x,pix_y] = (dot_product[pix_x,pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    

empty_rows = np.where(np.isnan(rgb_map_diff))
# Replace empty vectors with NaN values
rgb_map_diff[empty_rows] = 0
rgb_map_diff = np.where(mask_test == 1, rgb_map_diff, np.nan)
cmap = plt.get_cmap("jet")  
cmap.set_bad("black", alpha=1.0)  # Map NaN values to black

max_angle_difference = 180
# Calculate the percentage difference
rgb_map_diff_percentage = (rgb_map_diff / max_angle_difference) * 100

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff), cmap=cmap)
plt.colorbar()
plt.title("Angle Differences")
plt.axis('off')
plt.show()

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff_percentage), cmap=cmap)
plt.colorbar()
plt.title("Angle Differences %")
plt.axis('off')
plt.show()

rgb_map_diff2 = 1 - np.abs(rgb_map_diff3)

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff2), cmap=cmap)
plt.colorbar()
plt.title("Angle Differences cosθ = 1 - |rgb_nc . rgb_c|")
plt.axis('off')
plt.show()

max_angle_difference = 180
# Calculate the percentage difference
rgb_map_diff_percentage_ = (rgb_map_diff2 / max_angle_difference) * 100

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff_percentage_), cmap=cmap)
plt.colorbar()
plt.title("Angle Differences % cosθ")
plt.axis('off')
plt.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle('Results from fitting with LS python code abs diff', color='white', fontsize=25)

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(np.abs(np.nan_to_num(true_test_params_unvec_nc[:,:,4]* mask_test - true_test_params_unvec_c[:,:,4]* mask_test, nan=0))), cmap='gray')
plt.title('S0', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')


plt.subplot(2, 3, 2)
plt.imshow(np.rot90(np.abs(np.nan_to_num(true_test_params_unvec_nc[:,:,2]* mask_test - true_test_params_unvec_c[:,:,2]* mask_test, nan=0))), cmap='gray')
plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(np.abs(np.nan_to_num(true_test_params_unvec_nc[:,:,3]* mask_test - true_test_params_unvec_c[:,:,3]* mask_test, nan=0))), cmap='gray')
plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(np.abs(np.nan_to_num(FA_iter_nc - FA_iter_c, nan=0))), cmap='gray')
plt.title('FA map', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(np.abs(np.nan_to_num(MD_iter_nc - MD_iter_c, nan=0))), cmap='gray')
plt.title('MD [μm^2/ms]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_diff2), cmap='gray')
plt.title('cosθ = 1 - |rgb_nc . rgb_c| [%]', color='white',fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()                          
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
plt.setp(cbytick_obj, color='white')

plt.show()

