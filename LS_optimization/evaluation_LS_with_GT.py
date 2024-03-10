# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:09:10 2023

@author: pcastror

Visualize results from LS fitting

"""

from scipy.stats import circmean, circvar
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from scipy import stats
from dipy.reconst.dti import fractional_anisotropy
from dipy.io.image import load_nifti
import pandas as pd

correction = 'noGNL'
idx_image = '100307'

path_true_params = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_' + idx_image + '_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_' + idx_image
path_pred_params = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_' + idx_image + '_dwi_GNL_NO_correction_params.nii'
num = 69

################## GROUND TRUTH PARAMETERS #########################

with open('{}_mask.bin'.format(path_true_mask), 'rb') as file:
    # Load the object from the file
    file_loss_tissueval = os.path.basename(file.name)
    mask_test = pk.load(file)
    
print('mask shape: ', mask_test.shape)
mask_test = mask_test[:, :, num]

true_test_param2, affine = load_nifti(path_true_params)
true_test_params_unvec = true_test_param2[:, :, num, :]                                          # extract a specific slice
true_test_params_unvec[:,:,3] = true_test_params_unvec[:,:,2]*true_test_params_unvec[:,:,3]      # calculate d_per from k_per


true_test_params_vec_S = true_test_params_unvec[:, :, 4][mask_test == 1]
true_test_params_vec_AD = true_test_params_unvec[:, :, 2][mask_test == 1]
true_test_params_vec_RD = true_test_params_unvec[:, :, 3][mask_test == 1]
true_test_params_vec_ang1 = true_test_params_unvec[:, :, 0][mask_test == 1]
true_test_params_vec_ang2 = true_test_params_unvec[:, :, 1][mask_test == 1]


################## PREDICTED PARAMETERS #########################

loss_tissuetest, affine = load_nifti(path_pred_params)
loss_tissuetest[:, :, :, 3] = loss_tissuetest[:,  :, :, 2] * loss_tissuetest[:, :, :, 3]  # calculate d_per from k_per
param_name = ['Ang1', 'Ang2', 'AD', 'RD', 'S0']                   
loss_tissuetest = loss_tissuetest[:, :, num, :]                                           # extract a specific slice
print(loss_tissuetest.shape)

pred_test_params_vec_S = loss_tissuetest[:, :, 4][mask_test == 1]
pred_test_params_vec_AD = loss_tissuetest[:, :, 2][mask_test == 1]
pred_test_params_vec_RD = loss_tissuetest[:, :, 3][mask_test == 1]
pred_test_params_vec_ang1 = loss_tissuetest[:, :, 0][mask_test == 1]
pred_test_params_vec_ang2 = loss_tissuetest[:, :, 1][mask_test == 1]

print('PARAMETERS: ', param_name)

# Verify the size of the bin file
print('Shape of the loss_tissuetest (predicted parameters): ', loss_tissuetest.shape)
print('Number of samples loss_tissuetest: ', loss_tissuetest.shape[0])

parameters_tissuetest = loss_tissuetest.shape[1]


################## EVALUATE ANGLES #########################

print('')
print('EVALUATE ANGLES')

# true and predicted angles in radians ANG1
true_angles = true_test_params_vec_ang1
predicted_angles = pred_test_params_vec_ang1
predicted_angles = np.nan_to_num(predicted_angles, nan=0)

# Calculate circular mean and circular variance
mean_true = circmean(true_angles)
mean_predicted = circmean(predicted_angles)
variance_true = circvar(true_angles)
variance_predicted = circvar(predicted_angles)

print('')
print(f"True Circular Mean ang1: {mean_true} radians")
print(f"Predicted Circular Mean ang1: {mean_predicted} radians")
print(f"True Circular Variance ang1: {variance_true}")
print(f"Predicted Circular Variance ang1: {variance_predicted}")

# true and predicted angles in radians ANG2
true_angles = true_test_params_vec_ang2
predicted_angles = pred_test_params_vec_ang2
predicted_angles = np.nan_to_num(predicted_angles, nan=0)

# Calculate circular mean and circular variance
mean_true = circmean(true_angles)
mean_predicted = circmean(predicted_angles)
variance_true = circvar(true_angles)
variance_predicted = circvar(predicted_angles)

print('')
print(f"True Circular Mean ang2: {mean_true} radians")
print(f"Predicted Circular Mean ang2: {mean_predicted} radians")
print(f"True Circular Variance ang2: {variance_true}")
print(f"Predicted Circular Variance ang2: {variance_predicted}")

#### circular plots ####
angles_true = true_test_params_vec_ang1
angles_pred = pred_test_params_vec_ang1
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# Subplot 1 - Circular Plot of Directions true ang1
ax = axes[0]
ax = fig.add_subplot(121, polar=True)
ax.plot(angles_true, np.ones_like(angles_true), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Circular Plot of Directions true ang1")
# Subplot 2 - Circular Plot of Directions pred ang1
ax = axes[1]
ax = fig.add_subplot(122, polar=True)
ax.plot(angles_pred, np.ones_like(angles_pred), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Circular Plot of Directions pred ang1")
plt.tight_layout()
plt.show()

angles_true = true_test_params_vec_ang2
angles_pred = pred_test_params_vec_ang2
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
# Subplot 1 - Circular Plot of Directions true ang1
ax = axes[0]
ax = fig.add_subplot(121, polar=True)
ax.plot(angles_true, np.ones_like(angles_true), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Circular Plot of Directions true ang2")
# Subplot 2 - Circular Plot of Directions pred ang1
ax = axes[1]
ax = fig.add_subplot(122, polar=True)
ax.plot(angles_pred, np.ones_like(angles_pred), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Circular Plot of Directions pred ang2")
plt.tight_layout()
plt.show()

#################### EVALUATE GT IMAGE PREDICTIONS VS LS PREDICTIONS #########################

############### GT LS IMAGES ####################
axial_diffusivity = true_test_params_unvec[:, :, 2]
radial_diffusivity = true_test_params_unvec[:, :, 3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros(
    (axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_GT = fractional_anisotropy(diffusion_tensor) * mask_test
MD_GT = ((axial_diffusivity + radial_diffusivity +
           radial_diffusivity) / 3) * mask_test

# GT angles
# Define the size of your image
height = true_test_params_unvec.shape[0]
width = true_test_params_unvec.shape[1]

theta = true_test_params_unvec[:, :, 0]
phi = true_test_params_unvec[:, :, 1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_true = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_true[:, :, 0] = x_abs * 255   # Red channel
rgb_map_true[:, :, 1] = y_abs * 255   # Green channel
rgb_map_true[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_true[:, :, 0] = np.where(mask_test == 1, rgb_map_true[:, :, 0], np.nan)
rgb_map_true[:, :, 1] = np.where(mask_test == 1, rgb_map_true[:, :, 1], np.nan)
rgb_map_true[:, :, 2] = np.where(mask_test == 1, rgb_map_true[:, :, 2], np.nan)

rgb_map_true2 = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_true2[:, :, 0] = rgb_map_true[:, :, 0] * FA_GT   # Red channel
rgb_map_true2[:, :, 1] = rgb_map_true[:, :, 1] * FA_GT   # Green channel
rgb_map_true2[:, :, 2] = rgb_map_true[:, :, 2] * FA_GT   # Blue channel


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Results GT',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(true_test_params_unvec[:, :, 4]), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(true_test_params_unvec[:, :, 2]), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(true_test_params_unvec[:, :, 3]), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_GT), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_GT), cmap='gray')
plt.title('MD', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_true2), cmap='jet')
plt.title('rgb_map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()

############### PREDICTED LS IMAGES ####################

# Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
axial_diffusivity = loss_tissuetest[:, :, 2]
radial_diffusivity = loss_tissuetest[:, :, 3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros(
    (axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_pred = fractional_anisotropy(diffusion_tensor)
MD_pred = (loss_tissuetest[:, :, 2] + loss_tissuetest[:, :, 3] + loss_tissuetest[:, :, 3]) / 3

# predicted angles
height = loss_tissuetest.shape[0]
width = loss_tissuetest.shape[1]

theta = loss_tissuetest[:,:,0]
phi = loss_tissuetest[:,:,1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred[:, :, 0] = x_abs * 255   # Red channel
rgb_map_pred[:, :, 1] = y_abs * 255   # Green channel
rgb_map_pred[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_pred[:, :, 0] = np.where(mask_test == 1, rgb_map_pred[:, :, 0], np.nan)
rgb_map_pred[:, :, 1] = np.where(mask_test == 1, rgb_map_pred[:, :, 1], np.nan)
rgb_map_pred[:, :, 2] = np.where(mask_test == 1, rgb_map_pred[:, :, 2], np.nan)

rgb_map_pred2 = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred2[:, :, 0] = rgb_map_pred[:, :, 0] * FA_pred   # Red channel
rgb_map_pred2[:, :, 1] = rgb_map_pred[:, :, 1] * FA_pred   # Green channel
rgb_map_pred2[:, :, 2] = rgb_map_pred[:, :, 2] * FA_pred   # Blue channel


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Results from fitting with LS {correction}',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(loss_tissuetest[:, :, 4]), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(loss_tissuetest[:, :, 2]), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(loss_tissuetest[:, :, 3]), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred), cmap='gray')
plt.title('MD', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_pred), cmap='jet')
plt.title('rgb_map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Results from fitting with LS {correction}',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(loss_tissuetest[:, :, 4]), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(loss_tissuetest[:, :, 2]), cmap='gray', vmin=0, vmax=3.2)
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(loss_tissuetest[:, :, 3]), cmap='gray', vmin=0, vmax=3.2)
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred), cmap='gray', vmin=0, vmax=3.2)
plt.title('MD', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_pred2), cmap='jet')
plt.title('rgb_map * FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


####### COMPARE THEM ######

rgb_map_diff = np.empty((rgb_map_pred2.shape[0], rgb_map_pred2.shape[1]))
rgb_map_diff3 = np.empty((rgb_map_true2.shape[0], rgb_map_true2.shape[1]))

dot_product = np.empty((rgb_map_pred2.shape[0], rgb_map_pred2.shape[1]))
for pix_x in range(0, rgb_map_pred2.shape[0], 1):
    for pix_y in range(0, rgb_map_pred2.shape[1], 1):
        vector1 = np.array(rgb_map_pred2[pix_x, pix_y, :])
        vector1 = np.double(vector1)
        vector2 = np.array(rgb_map_true2[pix_x, pix_y, :])
        vector2 = np.double(vector2)
        dot_product[pix_x, pix_y] = np.dot(vector1, vector2)
        rgb_map_diff[pix_x, pix_y] = np.arccos(
            dot_product[pix_x, pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
        rgb_map_diff3[pix_x, pix_y] = (
            dot_product[pix_x, pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))

empty_rows = np.where(np.isnan(rgb_map_diff))
rgb_map_diff[empty_rows] = 0
rgb_map_diff = np.where(mask_test == 1, rgb_map_diff, np.nan)
cmap = plt.get_cmap("jet")
cmap.set_bad("black", alpha=1.0)  # Map NaN values to black

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff), cmap=cmap)
plt.colorbar()
plt.title(f'Angle Differences {correction}')
plt.axis('off')
plt.show()

rgb_map_diff2 = 1 - rgb_map_diff3


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'GT dataset {correction}',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(true_test_params_unvec[:, :, 4]), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(true_test_params_unvec[:, :, 2]), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(true_test_params_unvec[:, :, 3]), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_GT), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_GT), cmap='gray')
plt.title('MD', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_true2), cmap='jet')
plt.title('rgb_map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
fig.suptitle(f'Plot difference (GT - LS) {correction}', fontsize=25)

# Plot data in each subplot
im = axes[0, 0].imshow((np.rot90(
    loss_tissuetest[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test))), cmap='gray')
axes[0, 0].set_title('S0 diff', fontsize=15)
axes[0, 0].axis('off')
fig.colorbar(im, ax=axes[0, 0])

im = axes[0, 1].imshow(np.rot90(
    loss_tissuetest[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)), cmap='gray')
axes[0, 1].set_title('AD diff', fontsize=15)
axes[0, 1].axis('off')
fig.colorbar(im, ax=axes[0, 1])

im = axes[0, 2].imshow(np.rot90(
    loss_tissuetest[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)), cmap='gray')
axes[0, 2].set_title('RD diff', fontsize=15)
axes[0, 2].axis('off')
fig.colorbar(im, ax=axes[0, 2])

im = axes[1, 0].imshow(np.rot90(FA_pred - np.nan_to_num(FA_GT)), cmap='gray')
axes[1, 0].set_title('FA diff', fontsize=15)
axes[1, 0].axis('off')
fig.colorbar(im, ax=axes[1, 0])

im = axes[1, 1].imshow(np.rot90(MD_pred - np.nan_to_num(MD_GT)), cmap='gray')
axes[1, 1].set_title('MD diff', fontsize=15)
axes[1, 1].axis('off')
fig.colorbar(im, ax=axes[1, 1])

im = axes[1, 2].imshow(np.rot90(rgb_map_diff2), cmap=cmap)
axes[1, 2].set_title('rgb_map diff', fontsize=15)
axes[1, 2].axis('off')
fig.colorbar(im, ax=axes[1, 2])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Plot difference (GT - LS) {correction}',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(loss_tissuetest[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(loss_tissuetest[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(loss_tissuetest[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred - np.nan_to_num(FA_GT)), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred - np.nan_to_num(MD_GT)), cmap='gray')
plt.title('MD diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_diff2), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


S0_diff = loss_tissuetest[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)
AD_diff = loss_tissuetest[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)
RD_diff = loss_tissuetest[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)
FA_diff = FA_pred - np.nan_to_num(FA_GT)
MD_diff = MD_pred - np.nan_to_num(MD_GT)
Angles_diff = rgb_map_diff2


per = 99

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Plot difference (GT - LS) {correction}. {per}th percentile',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
S0_diff_flat = np.nan_to_num(S0_diff.flatten())
threshold_value = np.percentile(S0_diff_flat, per)
S0_diff_thresholded = np.clip(S0_diff, 0, threshold_value)
plt.imshow(np.rot90(S0_diff_thresholded), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
AD_diff_flat = np.nan_to_num(AD_diff.flatten())
threshold_value = np.percentile(AD_diff_flat, per)
AD_diff_thresholded = np.clip(AD_diff, 0, threshold_value)
plt.imshow(np.rot90(AD_diff_thresholded), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
RD_diff_flat = np.nan_to_num(RD_diff.flatten())
threshold_value = np.percentile(RD_diff_flat, per)
RD_diff_thresholded = np.clip(RD_diff, 0, threshold_value)
plt.imshow(np.rot90(RD_diff_thresholded), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
FA_diff_flat = np.nan_to_num(FA_diff.flatten())
threshold_value = np.percentile(FA_diff_flat, per)
FA_diff_thresholded = np.clip(FA_diff, 0, threshold_value)
plt.imshow(np.rot90(FA_diff_thresholded), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
MD_diff_flat = np.nan_to_num(MD_diff.flatten())
threshold_value = np.percentile(MD_diff_flat, per)
MD_diff_thresholded = np.clip(MD_diff, 0, threshold_value)
plt.imshow(np.rot90(MD_diff_thresholded), cmap='gray')
plt.title('MD diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
ang_diff_flat = np.nan_to_num(Angles_diff.flatten())
threshold_value = np.percentile(ang_diff_flat, per)
ang_diff_thresholded = np.clip(Angles_diff, 0, threshold_value)
plt.imshow(np.rot90(ang_diff_thresholded), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Plot difference (GT - LS) {correction}. [{num}]. idx: {idx_image}. {per}th per',
             fontsize=25, color='white')

plt.subplot(2, 3, 1)
S0_diff_flat = np.nan_to_num(S0_diff.flatten())
threshold_value = np.percentile(S0_diff_flat, per)
S0_diff_thresholded = np.clip(S0_diff, 0, threshold_value)
plt.imshow(np.rot90(S0_diff_thresholded), cmap='jet')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
AD_diff_flat = np.nan_to_num(AD_diff.flatten())
threshold_value = np.percentile(AD_diff_flat, per)
AD_diff_thresholded = np.clip(AD_diff, 0, threshold_value)
plt.imshow(np.rot90(AD_diff_thresholded), cmap='jet')
plt.title('AD/d_par [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
RD_diff_flat = np.nan_to_num(RD_diff.flatten())
threshold_value = np.percentile(RD_diff_flat, per)
RD_diff_thresholded = np.clip(RD_diff, 0, threshold_value)
plt.imshow(np.rot90(RD_diff_thresholded), cmap='jet')
plt.title('RD/d_per [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
FA_diff_flat = np.nan_to_num(FA_diff.flatten())
threshold_value = np.percentile(FA_diff_flat, per)
FA_diff_thresholded = np.clip(FA_diff, 0, threshold_value)
plt.imshow(np.rot90(FA_diff_thresholded), cmap='jet')
#plt.imshow(np.rot90(FA_diff_thresholded), cmap='coolwarm')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
MD_diff_flat = np.nan_to_num(MD_diff.flatten())
threshold_value = np.percentile(MD_diff_flat, per)
MD_diff_thresholded = np.clip(MD_diff, 0, threshold_value)
plt.imshow(np.rot90(MD_diff_thresholded), cmap='jet')
plt.title('MD diff [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
ang_diff_flat = np.nan_to_num(Angles_diff.flatten())
threshold_value = np.percentile(ang_diff_flat, per)
ang_diff_thresholded = np.clip(Angles_diff, 0, threshold_value)
plt.imshow(np.rot90(ang_diff_thresholded), cmap='jet')
plt.title('rgb_map diff. 1-|a.b|', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()




fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
fig.suptitle(f'Plot difference abs(GT - LS) {correction}', fontsize=25)

# Plot data in each subplot
im = axes[0, 0].imshow(np.rot90(np.abs(
    loss_tissuetest[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test))), cmap='gray')
axes[0, 0].set_title('S0 diff', fontsize=14)
axes[0, 0].axis('off')
fig.colorbar(im, ax=axes[0, 0])

im = axes[0, 1].imshow(np.rot90(np.abs(
    loss_tissuetest[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2]) * mask_test)), cmap='gray')
axes[0, 1].set_title('AD diff', fontsize=15)
axes[0, 1].axis('off')
fig.colorbar(im, ax=axes[0, 1])

im = axes[0, 2].imshow(np.rot90(np.abs(
    loss_tissuetest[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test))), cmap='gray')
axes[0, 2].set_title('RD diff', fontsize=15)
axes[0, 2].axis('off')
fig.colorbar(im, ax=axes[0, 2])

im = axes[1, 0].imshow(
    np.rot90(np.abs(FA_pred - np.nan_to_num(FA_GT))), cmap='gray')
axes[1, 0].set_title('FA diff', fontsize=15)
axes[1, 0].axis('off')
fig.colorbar(im, ax=axes[1, 0])

im = axes[1, 1].imshow(
    np.rot90(np.abs(MD_pred - np.nan_to_num(MD_GT))), cmap='gray')
axes[1, 1].set_title('MD diff', fontsize=15)
axes[1, 1].axis('off')
fig.colorbar(im, ax=axes[1, 1])

im = axes[1, 2].imshow(np.rot90(rgb_map_diff2), cmap=cmap, vmin = -1, vmax= 1)
axes[1, 2].set_title('rgb_map diff. 1-|a.b|', fontsize=15)
axes[1, 2].axis('off')
fig.colorbar(im, ax=axes[1, 2])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

S0_diff = loss_tissuetest[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)
AD_diff = loss_tissuetest[:, :, 2] - \
    np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)
RD_diff = loss_tissuetest[:, :, 3] - \
    np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)
FA_diff = FA_pred - np.nan_to_num(FA_GT)
MD_diff = MD_pred - np.nan_to_num(MD_GT)
Angles_diff = np.nan_to_num(rgb_map_diff2)

# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences {correction} [{num}]. idx: {idx_image}', fontsize=25)

S0_hist = abs(S0_diff[mask_test == 1])
axes[0, 0].hist(S0_hist, color='blue', bins=30)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
#axes[0, 0].legend()

d_par_hist = abs(AD_diff[mask_test == 1])
axes[0, 1].hist(d_par_hist, color='blue', bins=30)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_|| ')
axes[0, 1].set_ylabel('Frequency')
#axes[0, 1].legend()

d_per_hist = abs(RD_diff[mask_test == 1])
axes[0, 2].hist(d_per_hist, color='blue', bins=30)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
#axes[0, 2].legend()

FA_hist = abs(FA_diff[mask_test == 1])
axes[1, 0].hist(FA_hist, color='blue', bins=30)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()

MD_hist = abs(MD_diff[mask_test == 1])
axes[1, 1].hist(MD_hist, color='blue', bins=30)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
#axes[1, 1].legend()

ang2_nc_hist = abs(Angles_diff[mask_test == 1])
axes[1, 2].hist(ang2_nc_hist, color='blue', bins=30)
axes[1, 2].set_title('Angles')
axes[1, 2].set_xlabel('Angles [º]')
axes[1, 2].set_ylabel('Frequency')
#axes[1, 2].legend()

plt.tight_layout()
plt.show()


# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences {correction} [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
axes[0, 0].hist(S0_hist_thresholded, color='blue', bins=30)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
#axes[0, 0].legend()

d_par_hist = abs(AD_diff[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
axes[0, 1].hist(AD_hist_thresholded, color='blue', bins=30)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
#axes[0, 1].legend()

d_per_hist = abs(RD_diff[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
axes[0, 2].hist(RD_hist_thresholded, color='blue', bins=30)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
#axes[0, 2].legend()

FA_hist = abs(FA_diff[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
axes[1, 0].hist(FA_hist_thresholded, color='blue', bins=30)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()

MD_hist = abs(MD_diff[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
axes[1, 1].hist(MD_hist_thresholded, color='blue', bins=30)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
#axes[1, 1].legend()

ang_hist = abs(Angles_diff[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
axes[1, 2].hist(ang_hist_thresholded, color='blue', bins=30)
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
#axes[1, 2].legend()

plt.tight_layout()
plt.show()



###################################################################################################

correction = 'GNL'
idx_image = '100307'

path_pred_params2 = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_' + idx_image + '_dwi_GNL_WITH_correction_params.nii'

################## PREDICTED PARAMETERS #########################

loss_tissuetest2, affine = load_nifti(path_pred_params2)
loss_tissuetest2[:, :, :, 3] = loss_tissuetest2[:,  :, :, 2] * loss_tissuetest2[:, :, :, 3]  # calculate d_per from k_per
loss_tissuetest2 = loss_tissuetest2[:, :, num, :]                                           # extract a specific slice
print(loss_tissuetest2.shape)

pred_test_params_vec_S2 = loss_tissuetest2[:, :, 4][mask_test == 1]
pred_test_params_vec_AD2 = loss_tissuetest2[:, :, 2][mask_test == 1]
pred_test_params_vec_RD2 = loss_tissuetest2[:, :, 3][mask_test == 1]
pred_test_params_vec_ang12 = loss_tissuetest2[:, :, 0][mask_test == 1]
pred_test_params_vec_ang22 = loss_tissuetest2[:, :, 1][mask_test == 1]

# # Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
axial_diffusivity = loss_tissuetest2[:, :, 2]
radial_diffusivity = loss_tissuetest2[:, :, 3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros(
    (axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_pred2 = fractional_anisotropy(diffusion_tensor)
MD_pred2 = (loss_tissuetest2[:, :, 2] + loss_tissuetest2[:, :, 3] + loss_tissuetest2[:, :, 3]) / 3

# predicted angles
height = loss_tissuetest2.shape[0]
width = loss_tissuetest2.shape[1]

theta = loss_tissuetest2[:,:,0]
phi = loss_tissuetest2[:,:,1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred[:, :, 0] = x_abs * 255   # Red channel
rgb_map_pred[:, :, 1] = y_abs * 255   # Green channel
rgb_map_pred[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_pred[:, :, 0] = np.where(mask_test == 1, rgb_map_pred[:, :, 0], np.nan)
rgb_map_pred[:, :, 1] = np.where(mask_test == 1, rgb_map_pred[:, :, 1], np.nan)
rgb_map_pred[:, :, 2] = np.where(mask_test == 1, rgb_map_pred[:, :, 2], np.nan)

rgb_map_pred22 = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred22[:, :, 0] = rgb_map_pred[:, :, 0] * FA_pred2   # Red channel
rgb_map_pred22[:, :, 1] = rgb_map_pred[:, :, 1] * FA_pred2   # Green channel
rgb_map_pred22[:, :, 2] = rgb_map_pred[:, :, 2] * FA_pred2   # Blue channel


rgb_map_diff = np.empty((rgb_map_pred22.shape[0], rgb_map_pred22.shape[1]))
rgb_map_diff3 = np.empty((rgb_map_true2.shape[0], rgb_map_true2.shape[1]))

dot_product = np.empty((rgb_map_pred22.shape[0], rgb_map_pred22.shape[1]))
for pix_x in range(0, rgb_map_pred22.shape[0], 1):
    for pix_y in range(0, rgb_map_pred22.shape[1], 1):
        vector1 = np.array(rgb_map_pred22[pix_x, pix_y, :])
        vector1 = np.double(vector1)
        vector2 = np.array(rgb_map_true2[pix_x, pix_y, :])
        vector2 = np.double(vector2)
        dot_product[pix_x, pix_y] = np.dot(vector1, vector2)
        rgb_map_diff[pix_x, pix_y] = np.arccos(
            dot_product[pix_x, pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
        rgb_map_diff3[pix_x, pix_y] = (
            dot_product[pix_x, pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))

empty_rows = np.where(np.isnan(rgb_map_diff))
rgb_map_diff[empty_rows] = 0
rgb_map_diff = np.where(mask_test == 1, rgb_map_diff, np.nan)
cmap = plt.get_cmap("jet")
cmap.set_bad("black", alpha=1.0)  # Map NaN values to black

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff), cmap=cmap)
plt.colorbar()
plt.title(f'Angle Differences {correction}')
plt.axis('off')
plt.show()

rgb_map_diff22 = 1 - rgb_map_diff3

S0_diff2 = loss_tissuetest2[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)
AD_diff2 = loss_tissuetest2[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)
RD_diff2 = loss_tissuetest2[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)
FA_diff2 = FA_pred2 - np.nan_to_num(FA_GT)
MD_diff2 = MD_pred2 - np.nan_to_num(MD_GT)
Angles_diff2 = np.nan_to_num(rgb_map_diff22)

# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences together. [{num}]. idx: {idx_image}.', fontsize=20)

S0_hist = abs(S0_diff[mask_test == 1])
S0_hist2 = abs(S0_diff2[mask_test == 1])
axes[0, 0].hist(S0_hist, color='blue', bins=30, label='no GNL')
axes[0, 0].hist(S0_hist2, color='red', bins=30, label='GNL')
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

d_par_hist = abs(AD_diff[mask_test == 1])
d_par_hist2 = abs(AD_diff2[mask_test == 1])
axes[0, 1].hist(d_par_hist, color='blue', bins=30, label='no GNL')
axes[0, 1].hist(d_par_hist2, color='red', bins=30, label='GNL')
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

d_per_hist = abs(RD_diff[mask_test == 1])
d_per_hist2 = abs(RD_diff2[mask_test == 1])
axes[0, 2].hist(d_per_hist, color='blue', bins=30, label='no GNL')
axes[0, 2].hist(d_per_hist2, color='red', bins=30, label='GNL')
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

FA_hist = abs(FA_diff[mask_test == 1])
FA_hist2 = abs(FA_diff2[mask_test == 1])
axes[1, 0].hist(FA_hist, color='blue', bins=30, label='no GNL')
axes[1, 0].hist(FA_hist2, color='red', bins=30, label='GNL')
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

MD_hist = abs(MD_diff[mask_test == 1])
MD_hist2 = abs(MD_diff2[mask_test == 1])
axes[1, 1].hist(MD_hist, color='blue', bins=30, label='no GNL')
axes[1, 1].hist(MD_hist2, color='red', bins=30, label='GNL')
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

ang_hist = abs(Angles_diff[mask_test == 1])
ang_hist2 = abs(Angles_diff2[mask_test == 1])
axes[1, 2].hist(ang_hist, color='blue', bins=30, label='no GNL')
axes[1, 2].hist(ang_hist2, color='red', bins=30, label='GNL')
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences together. [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
S0_hist2 = abs(S0_diff2[mask_test == 1])
threshold_value = np.percentile(S0_hist2, per)
S0_hist_thresholded2 = np.clip(S0_hist2, 0, threshold_value)
axes[0, 0].hist(S0_hist_thresholded, color='blue', bins=30, label='no GNL')
axes[0, 0].hist(S0_hist_thresholded2, color='red', bins=30, label='GNL')
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

d_par_hist = abs(AD_diff[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
d_par_hist2 = abs(AD_diff2[mask_test == 1])
threshold_value = np.percentile(d_par_hist2, per)
AD_hist_thresholded2 = np.clip(d_par_hist2, 0, threshold_value)
axes[0, 1].hist(AD_hist_thresholded, color='blue', bins=30, label='no GNL')
axes[0, 1].hist(AD_hist_thresholded2, color='red', bins=30, label='GNL')
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

d_per_hist = abs(RD_diff[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
d_per_hist2 = abs(RD_diff2[mask_test == 1])
threshold_value = np.percentile(d_per_hist2, per)
RD_hist_thresholded2 = np.clip(d_per_hist2, 0, threshold_value)
axes[0, 2].hist(RD_hist_thresholded, color='blue', bins=30, label='no GNL')
axes[0, 2].hist(RD_hist_thresholded2, color='red', bins=30, label='GNL')
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

FA_hist = abs(FA_diff[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
FA_hist2 = abs(FA_diff2[mask_test == 1])
threshold_value = np.percentile(FA_hist2, per)
FA_hist_thresholded2 = np.clip(FA_hist2, 0, threshold_value)
axes[1, 0].hist(FA_hist_thresholded, color='blue', bins=30, label='no GNL')
axes[1, 0].hist(FA_hist_thresholded2, color='red', bins=30, label='GNL')
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

MD_hist = abs(MD_diff[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
MD_hist2 = abs(MD_diff2[mask_test == 1])
threshold_value = np.percentile(MD_hist2, per)
MD_hist_thresholded2 = np.clip(MD_hist2, 0, threshold_value)
axes[1, 1].hist(MD_hist_thresholded, color='blue', bins=30, label='no GNL')
axes[1, 1].hist(MD_hist_thresholded2, color='red', bins=30, label='GNL')
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

ang_hist = abs(Angles_diff[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
ang_hist2 = abs(Angles_diff2[mask_test == 1])
threshold_value = np.percentile(ang_hist2, per)
ang_hist_thresholded2 = np.clip(ang_hist2, 0, threshold_value)
axes[1, 2].hist(ang_hist_thresholded, color='blue', bins=30, label='no GNL')
axes[1, 2].hist(ang_hist_thresholded2, color='red', bins=30, label='GNL')
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences together. [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
S0_hist2 = abs(S0_diff2[mask_test == 1])
threshold_value = np.percentile(S0_hist2, per)
S0_hist_thresholded2 = np.clip(S0_hist2, 0, threshold_value)
axes[0, 0].hist(S0_hist_thresholded, color='blue', label='no GNL')
axes[0, 0].hist(S0_hist_thresholded2, color='red', label='GNL')
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

d_par_hist = abs(AD_diff[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
d_par_hist2 = abs(AD_diff2[mask_test == 1])
threshold_value = np.percentile(d_par_hist2, per)
AD_hist_thresholded2 = np.clip(d_par_hist2, 0, threshold_value)
axes[0, 1].hist(AD_hist_thresholded, color='blue',  label='no GNL')
axes[0, 1].hist(AD_hist_thresholded2, color='red',label='GNL')
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

d_per_hist = abs(RD_diff[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
d_per_hist2 = abs(RD_diff2[mask_test == 1])
threshold_value = np.percentile(d_per_hist2, per)
RD_hist_thresholded2 = np.clip(d_per_hist2, 0, threshold_value)
axes[0, 2].hist(RD_hist_thresholded, color='blue', label='no GNL')
axes[0, 2].hist(RD_hist_thresholded2, color='red', label='GNL')
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

FA_hist = abs(FA_diff[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
FA_hist2 = abs(FA_diff2[mask_test == 1])
threshold_value = np.percentile(FA_hist2, per)
FA_hist_thresholded2 = np.clip(FA_hist2, 0, threshold_value)
axes[1, 0].hist(FA_hist_thresholded, color='blue',  label='no GNL')
axes[1, 0].hist(FA_hist_thresholded2, color='red',  label='GNL')
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

MD_hist = abs(MD_diff[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
MD_hist2 = abs(MD_diff2[mask_test == 1])
threshold_value = np.percentile(MD_hist2, per)
MD_hist_thresholded2 = np.clip(MD_hist2, 0, threshold_value)
axes[1, 1].hist(MD_hist_thresholded, color='blue', label='no GNL')
axes[1, 1].hist(MD_hist_thresholded2, color='red',  label='GNL')
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

ang_hist = abs(Angles_diff[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
ang_hist2 = abs(Angles_diff2[mask_test == 1])
threshold_value = np.percentile(ang_hist2, per)
ang_hist_thresholded2 = np.clip(ang_hist2, 0, threshold_value)
axes[1, 2].hist(ang_hist_thresholded, color='blue', label='no GNL')
axes[1, 2].hist(ang_hist_thresholded2, color='red', label='GNL')
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()

plt.tight_layout()
plt.show()




###############################################################################################################3


loss_tissuetest_ = loss_tissuetest[:][mask_test == 1]

# dot product

# true angles
height = true_test_params_vec_S.shape[0]

theta = true_test_params_vec_ang1
phi = true_test_params_vec_ang2

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_true = np.zeros((height, 3), dtype=np.uint8)
rgb_map_true[:, 0] = x_abs * 255   # Red channel
rgb_map_true[:, 1] = y_abs * 255   # Green channel
rgb_map_true[:, 2] = z_abs * 255   # Blue channel

# predicted angles
height = pred_test_params_vec_S.shape[0]

theta = pred_test_params_vec_ang1
phi = pred_test_params_vec_ang2

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred = np.zeros((height, 3), dtype=np.uint8)
rgb_map_pred[:, 0] = x_abs * 255   # Red channel
rgb_map_pred[:, 1] = y_abs * 255   # Green channel
rgb_map_pred[:, 2] = z_abs * 255   # Blue channel


dot_product = np.empty((rgb_map_true.shape[0]))
for x in range(0, rgb_map_true.shape[0], 1):
    vector1 = np.array(rgb_map_true[x, :])
    vector1 = np.double(vector1)/np.linalg.norm(vector1)
    vector2 = np.array(rgb_map_pred[x, :])
    vector2 = np.double(vector2)/np.linalg.norm(vector2)
    dot_product[x] = round(np.dot(vector1, vector2), 2)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.suptitle(
    f'Scatter plot Ground truth vs LS prediction no GNL. [{num}]. idx: {idx_image}', fontsize=25)

axes[0].hist(1 - abs(dot_product), color='b')
axes[0].set_xlabel('1 - |a . b|')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Angles')

axes[1].scatter(true_test_params_vec_AD, loss_tissuetest_[:, 2], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_AD,  loss_tissuetest_[:, 2])
trend_line = slope * np.array(true_test_params_vec_AD) + intercept
axes[1].plot(true_test_params_vec_AD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[1].set_xlabel('True AD')
axes[1].set_ylabel('Predicted AD')
axes[1].set_title('AD')

axes[2].scatter(true_test_params_vec_RD, loss_tissuetest_[:, 3], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_RD,  loss_tissuetest_[:, 3])
trend_line = slope * np.array(true_test_params_vec_RD) + intercept
axes[2].plot(true_test_params_vec_RD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[2].set_xlabel('True RD')
axes[2].set_ylabel('Predicted RD')
axes[2].set_title('RD')

axes[3].scatter(true_test_params_vec_S, loss_tissuetest_[:, 4], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_S,  loss_tissuetest_[:, 4])
trend_line = slope * np.array(true_test_params_vec_S) + intercept
axes[3].plot(true_test_params_vec_S, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[3].set_xlabel('True S0')
axes[3].set_ylabel('Predicted S0')
axes[3].set_title('S0')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.suptitle(
    f'Scatter plot Ground truth vs LS prediction no GNL. [{num}]. idx: {idx_image}', fontsize=25)

axes[0].hist(1 - abs(dot_product), color='b')
axes[0].set_xlabel('1 - |a . b|')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Angles')

axes[1].scatter(true_test_params_vec_AD, loss_tissuetest_[:, 2], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_AD,  loss_tissuetest_[:, 2])
trend_line = slope * np.array(true_test_params_vec_AD) + intercept
axes[1].plot(true_test_params_vec_AD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[1].set_xlabel('True AD')
axes[1].set_ylabel('Predicted AD')
axes[1].set_title('AD')

axes[2].scatter(true_test_params_vec_RD/true_test_params_vec_AD, loss_tissuetest_[:, 3]/loss_tissuetest_[:, 2], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_RD/true_test_params_vec_AD,  loss_tissuetest_[:, 3]/loss_tissuetest_[:, 2])
trend_line = slope * np.array(true_test_params_vec_RD/true_test_params_vec_AD) + intercept
axes[2].plot(true_test_params_vec_RD/true_test_params_vec_AD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[2].set_xlabel('True kper')
axes[2].set_ylabel('Predicted kper')
axes[2].set_title('kper')

axes[3].scatter(true_test_params_vec_S, loss_tissuetest_[:, 4], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_S,  loss_tissuetest_[:, 4])
trend_line = slope * np.array(true_test_params_vec_S) + intercept
axes[3].plot(true_test_params_vec_S, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[3].set_xlabel('True S0')
axes[3].set_ylabel('Predicted S0')
axes[3].set_title('S0')

plt.tight_layout()
plt.show()






# Creating the DataFrame
statistics_table = {
    'AD GT': true_test_params_vec_AD,
    'AD LS': pred_test_params_vec_AD
}

df_stats = pd.DataFrame(statistics_table)
#print(df_stats.to_string(index=False))

