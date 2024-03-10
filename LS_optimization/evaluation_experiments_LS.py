# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:07:54 2023

@author: pcastror

EVALUATION LS WITH GT. COMPARISON OF DATASET FITTING INCLUDING AND NOT INCLUDDING GNL CORRECTION
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
import seaborn as sns


idx_image = '100307'

path_true_params = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_' + idx_image + '_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_' + idx_image
path_pred_params_no_correction = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_' + idx_image + '_dwi_GNL_NO_correction_params.nii'
path_pred_params_correction = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_' + idx_image + '_dwi_GNL_WITH_correction_params.nii'
num = 70  # slice number to plot
param_name = ['Ang1', 'Ang2', 'AD', 'RD', 'S0']                   
print('PARAMETERS: ', param_name)
per = 99

################## LOAD GROUND TRUTH PARAMETERS #########################

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


################## LOAD PREDICTED PARAMETERS WITHOUT GNL CORRECTIONG DURING FITTING #########################

pred_test_param_no_GNL, affine = load_nifti(path_pred_params_no_correction)
pred_test_param_no_GNL[:, :, :, 3] = pred_test_param_no_GNL[:,  :, :, 2] * pred_test_param_no_GNL[:, :, :, 3]  # calculate d_per from k_per
pred_test_param_no_GNL = pred_test_param_no_GNL[:, :, num, :]     # extract slice                                        # extract a specific slice
print('Shape pred_test_param_no_GNL: ', pred_test_param_no_GNL.shape)

pred_test_params_no_GNL_vec_S = pred_test_param_no_GNL[:, :, 4][mask_test == 1]
pred_test_params_no_GNL_vec_AD = pred_test_param_no_GNL[:, :, 2][mask_test == 1]
pred_test_params_no_GNL_vec_RD = pred_test_param_no_GNL[:, :, 3][mask_test == 1]
pred_test_params_no_GNL_vec_ang1 = pred_test_param_no_GNL[:, :, 0][mask_test == 1]
pred_test_params_no_GNL_vec_ang2 = pred_test_param_no_GNL[:, :, 1][mask_test == 1]


################## LOAD PREDICTED PARAMETERS WITHOUT GNL CORRECTIONG DURING FITTING #########################

pred_test_param_GNL, affine = load_nifti(path_pred_params_correction)
pred_test_param_GNL[:, :, :, 3] = pred_test_param_GNL[:,  :, :, 2] * pred_test_param_GNL[:, :, :, 3]  # calculate d_per from k_per
pred_test_param_GNL = pred_test_param_GNL[:, :, num, :]     # extract slice                                        # extract a specific slice
print('Shape pred_test_param_GNL: ', pred_test_param_GNL.shape)

pred_test_params_GNL_vec_S = pred_test_param_GNL[:, :, 4][mask_test == 1]
pred_test_params_GNL_vec_AD = pred_test_param_GNL[:, :, 2][mask_test == 1]
pred_test_params_GNL_vec_RD = pred_test_param_GNL[:, :, 3][mask_test == 1]
pred_test_params_GNL_vec_ang1 = pred_test_param_GNL[:, :, 0][mask_test == 1]
pred_test_params_GNL_vec_ang2 = pred_test_param_GNL[:, :, 1][mask_test == 1]



############## START EVALUATING###################


################## EVALUATE ANGLES #########################

print('')
print('EVALUATE ANGLES')

# true and predicted angles in radians ANG1
true_angles = true_test_params_vec_ang1                                # true angles
predicted_angles_no_GNL = pred_test_params_no_GNL_vec_ang1             # predicted no correction
predicted_angles_no_GNL = np.nan_to_num(predicted_angles_no_GNL, nan=0)
predicted_angles_GNL = pred_test_params_GNL_vec_ang1                   # predicted with correction
predicted_angles_GNL = np.nan_to_num(predicted_angles_GNL, nan=0)

# Calculate circular mean and circular variance
mean_true = circmean(true_angles)
mean_predicted_no_GNL = circmean(predicted_angles_no_GNL)
mean_predicted_GNL = circmean(predicted_angles_GNL)

variance_true = circvar(true_angles)
variance_predicted_no_GNL = circvar(predicted_angles_no_GNL)
variance_predicted_GNL = circvar(predicted_angles_GNL)

print('')
print(f"True Circular Mean ang1: {mean_true} radians")
print(f"Predicted Circular Mean ang1 no GNL correction: {mean_predicted_no_GNL} radians")
print(f"Predicted Circular Mean ang1 with GNL correction: {mean_predicted_GNL} radians")

print(f"True Circular Variance ang1: {variance_true}")
print(f"Predicted Circular Variance ang1 no GNL correction: {variance_predicted_no_GNL}")
print(f"Predicted Circular Variance ang1 with GNL correction: {variance_predicted_GNL}")

# true and predicted angles in radians ANG2
true_angles = true_test_params_vec_ang2                                # true angles
predicted_angles_no_GNL = pred_test_params_no_GNL_vec_ang2             # predicted no correction
predicted_angles_no_GNL = np.nan_to_num(predicted_angles_no_GNL, nan=0)
predicted_angles_GNL = pred_test_params_GNL_vec_ang2                  # predicted with correction
predicted_angles_GNL = np.nan_to_num(predicted_angles_GNL, nan=0)

# Calculate circular mean and circular variance
mean_true = circmean(true_angles)
mean_predicted_no_GNL = circmean(predicted_angles_no_GNL)
mean_predicted_GNL = circmean(predicted_angles_GNL)

variance_true = circvar(true_angles)
variance_predicted_no_GNL = circvar(predicted_angles_no_GNL)
variance_predicted_GNL = circvar(predicted_angles_GNL)

print('')
print(f"True Circular Mean ang2: {mean_true} radians")
print(f"Predicted Circular Mean ang2 no GNL correction: {mean_predicted_no_GNL} radians")
print(f"Predicted Circular Mean ang2 with GNL correction: {mean_predicted_GNL} radians")

print(f"True Circular Variance ang2: {variance_true}")
print(f"Predicted Circular Variance ang2 no GNL correction: {variance_predicted_no_GNL}")
print(f"Predicted Circular Variance ang2 with GNL correction: {variance_predicted_GNL}")

#### circular plots ####
angles_true = true_test_params_vec_ang1
angles_pred_no_GNL = pred_test_params_no_GNL_vec_ang1
angles_pred_GNL = pred_test_params_GNL_vec_ang1

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Circular Plot of Directions Ang1', fontsize=16)
# true ang1
ax = axes[0]
ax = fig.add_subplot(131, polar=True)
ax.plot(angles_true, np.ones_like(angles_true), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("True", fontsize=16)
# pred ang1 no GNL correction
ax = axes[1]
ax = fig.add_subplot(132, polar=True)
ax.plot(angles_pred_no_GNL, np.ones_like(angles_pred_no_GNL), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Predicted no GNL correction", fontsize=16)
plt.tight_layout()
# Circular Plot of Directions pred ang1 with GNL correction
ax = axes[2]
ax = fig.add_subplot(133, polar=True)
ax.plot(angles_pred_GNL, np.ones_like(angles_pred_GNL), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Predicted with GNL correction", fontsize=16)
plt.tight_layout()
plt.show()

angles_true = true_test_params_vec_ang2
angles_pred_no_GNL = pred_test_params_no_GNL_vec_ang2
angles_pred_GNL = pred_test_params_GNL_vec_ang2

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Circular Plot of Directions Ang2', fontsize=16)
# true ang1
ax = axes[0]
ax = fig.add_subplot(131, polar=True)
ax.plot(angles_true, np.ones_like(angles_true), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("True", fontsize=16)
# pred ang1 no GNL correction
ax = axes[1]
ax = fig.add_subplot(132, polar=True)
ax.plot(angles_pred_no_GNL, np.ones_like(angles_pred_no_GNL), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Predicted no GNL correction", fontsize=16)
plt.tight_layout()
# Circular Plot of Directions pred ang1 with GNL correction
ax = axes[2]
ax = fig.add_subplot(133, polar=True)
ax.plot(angles_pred_GNL, np.ones_like(angles_pred_GNL), 'ro', markersize=8)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(90)
ax.grid(True)
ax.set_title("Predicted with GNL correction", fontsize=16)
plt.tight_layout()
plt.show()

#################### EVALUATE GT IMAGE PREDICTIONS VS LS PREDICTIONS #########################

############### GT IMAGES ####################
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

##### no GNL correction #####

# Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
axial_diffusivity = pred_test_param_no_GNL[:, :, 2]
radial_diffusivity = pred_test_param_no_GNL[:, :, 3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_pred_no_GNL = fractional_anisotropy(diffusion_tensor)
MD_pred_no_GNL = (pred_test_param_no_GNL[:, :, 2] + pred_test_param_no_GNL[:, :, 3] + pred_test_param_no_GNL[:, :, 3]) / 3

# predicted angles
height = pred_test_param_no_GNL.shape[0]
width = pred_test_param_no_GNL.shape[1]

theta = pred_test_param_no_GNL[:,:,0]
phi = pred_test_param_no_GNL[:,:,1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred_no_GNL = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred_no_GNL[:, :, 0] = x_abs * 255   # Red channel
rgb_map_pred_no_GNL[:, :, 1] = y_abs * 255   # Green channel
rgb_map_pred_no_GNL[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_pred_no_GNL[:, :, 0] = np.where(mask_test == 1, rgb_map_pred_no_GNL[:, :, 0], np.nan)
rgb_map_pred_no_GNL[:, :, 1] = np.where(mask_test == 1, rgb_map_pred_no_GNL[:, :, 1], np.nan)
rgb_map_pred_no_GNL[:, :, 2] = np.where(mask_test == 1, rgb_map_pred_no_GNL[:, :, 2], np.nan)

rgb_map_pred2_no_GNL = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred2_no_GNL[:, :, 0] = rgb_map_pred_no_GNL[:, :, 0] * FA_pred_no_GNL   # Red channel
rgb_map_pred2_no_GNL[:, :, 1] = rgb_map_pred_no_GNL[:, :, 1] * FA_pred_no_GNL   # Green channel
rgb_map_pred2_no_GNL[:, :, 2] = rgb_map_pred_no_GNL[:, :, 2] * FA_pred_no_GNL   # Blue channel

fig, axes = plt.subplots(1, 2, figsize=(16, 8) , facecolor='black')
fig.suptitle('Angles no GNL correction', color='white')
im1 = axes[0].imshow(np.rot90(rgb_map_pred_no_GNL), cmap='jet')
axes[0].set_title('rgb_map', color='white', fontsize=20)
axes[0].axis('off')
# =============================================================================
# colorbar1 = plt.colorbar(im1, ax=axes[0])
# colorbar1.ax.yaxis.set_tick_params(color='white',  labelsize=20)
# =============================================================================
im2 = axes[1].imshow(np.rot90(rgb_map_pred2_no_GNL), cmap='jet')
axes[1].set_title('rgb_map * FA map', color='white', fontsize=20)
axes[1].axis('off')
# =============================================================================
# colorbar2 = plt.colorbar(im2, ax=axes[1])
# colorbar2.ax.yaxis.set_tick_params(color='white',  labelsize=15)
# =============================================================================
fig.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Results from fitting with LS without GNL corretion. Slice {[num]}', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(pred_test_param_no_GNL[:, :, 4]), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(pred_test_param_no_GNL[:, :, 2]), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(pred_test_param_no_GNL[:, :, 3]), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred_no_GNL), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred_no_GNL), cmap='gray')
plt.title('MD', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_pred2_no_GNL), cmap='jet')
plt.title('rgb_map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()

rgb_map_diff = np.empty((rgb_map_pred2_no_GNL.shape[0], rgb_map_pred2_no_GNL.shape[1]))
rgb_map_diff3 = np.empty((rgb_map_true2.shape[0], rgb_map_true2.shape[1]))

dot_product = np.empty((rgb_map_pred2_no_GNL.shape[0], rgb_map_pred2_no_GNL.shape[1]))
for pix_x in range(0, rgb_map_pred2_no_GNL.shape[0], 1):
    for pix_y in range(0, rgb_map_pred2_no_GNL.shape[1], 1):
        vector1 = np.array(rgb_map_pred2_no_GNL[pix_x, pix_y, :])
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
rgb_map_diff_no_GNL = np.where(mask_test == 1, rgb_map_diff, np.nan)
cmap = plt.get_cmap("jet")
cmap.set_bad("black", alpha=1.0)  # Map NaN values to black

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff_no_GNL), cmap=cmap)
plt.colorbar()
plt.title(f'Angle Differences no GNL correction')
plt.axis('off')
plt.show()

rgb_map_diff2_no_GNL = 1 - rgb_map_diff3

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff2_no_GNL), cmap=cmap)
plt.colorbar()
plt.title(f'Angle Differences no GNL correction. 1-diff')
plt.axis('off')
plt.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle('Plot difference (GT - LS) no GNL correction', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(pred_test_param_no_GNL[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(pred_test_param_no_GNL[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(pred_test_param_no_GNL[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred_no_GNL - np.nan_to_num(FA_GT)), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred_no_GNL - np.nan_to_num(MD_GT)), cmap='gray')
plt.title('MD diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_diff2_no_GNL), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()



S0_diff_no_GNL = pred_test_param_no_GNL[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)
AD_diff_no_GNL = pred_test_param_no_GNL[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)
RD_diff_no_GNL = pred_test_param_no_GNL[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)
FA_diff_no_GNL = FA_pred_no_GNL - np.nan_to_num(FA_GT)
MD_diff_no_GNL = MD_pred_no_GNL - np.nan_to_num(MD_GT)
Angles_diff_no_GNL = rgb_map_diff2_no_GNL

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Plot difference (GT - LS) no GNL correction. percentile{per}', fontsize=25, color='white')

plt.subplot(2, 3, 1)
S0_diff_flat = np.nan_to_num(S0_diff_no_GNL.flatten())
threshold_value = np.percentile(S0_diff_flat, per)
S0_diff_thresholded_no_GNL = np.clip(S0_diff_no_GNL, 0, threshold_value)
plt.imshow(np.rot90(S0_diff_thresholded_no_GNL), cmap='jet')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
AD_diff_flat = np.nan_to_num(AD_diff_no_GNL.flatten())
threshold_value = np.percentile(AD_diff_flat, per)
AD_diff_thresholded_no_GNL = np.clip(AD_diff_no_GNL, 0, threshold_value)
plt.imshow(np.rot90(AD_diff_thresholded_no_GNL), cmap='jet')
plt.title('AD/d_par [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
RD_diff_flat = np.nan_to_num(RD_diff_no_GNL.flatten())
threshold_value = np.percentile(RD_diff_flat, per)
RD_diff_thresholded_no_GNL = np.clip(RD_diff_no_GNL, 0, threshold_value)
plt.imshow(np.rot90(RD_diff_thresholded_no_GNL), cmap='jet')
plt.title('RD/d_per [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
FA_diff_flat = np.nan_to_num(FA_diff_no_GNL.flatten())
threshold_value = np.percentile(FA_diff_flat, per)
FA_diff_thresholded_no_GNL = np.clip(FA_diff_no_GNL, 0, threshold_value)
plt.imshow(np.rot90(FA_diff_thresholded_no_GNL), cmap='jet')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
MD_diff_flat = np.nan_to_num(MD_diff_no_GNL.flatten())
threshold_value = np.percentile(MD_diff_flat, per)
MD_diff_thresholded_no_GNL = np.clip(MD_diff_no_GNL, 0, threshold_value)
plt.imshow(np.rot90(MD_diff_thresholded_no_GNL), cmap='jet')
plt.title('MD diff [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
ang_diff_flat = np.nan_to_num(Angles_diff_no_GNL.flatten())
threshold_value = np.percentile(ang_diff_flat, per)
ang_diff_thresholded_no_GNL = np.clip(Angles_diff_no_GNL, 0, threshold_value)
plt.imshow(np.rot90(ang_diff_thresholded_no_GNL), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()

# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences no GNL correction [{num}]. idx: {idx_image}', fontsize=25)

S0_hist = abs(S0_diff_no_GNL[mask_test == 1])
axes[0, 0].hist(S0_hist, color='blue', bins=30)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
#axes[0, 0].legend()

d_par_hist = abs(AD_diff_no_GNL[mask_test == 1])
axes[0, 1].hist(d_par_hist, color='blue', bins=30)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_|| ')
axes[0, 1].set_ylabel('Frequency')
#axes[0, 1].legend()

d_per_hist = abs(RD_diff_no_GNL[mask_test == 1])
axes[0, 2].hist(d_per_hist, color='blue', bins=30)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
#axes[0, 2].legend()

FA_hist = abs(FA_diff_no_GNL[mask_test == 1])
axes[1, 0].hist(FA_hist, color='blue', bins=30)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()

MD_hist = abs(MD_diff_no_GNL[mask_test == 1])
axes[1, 1].hist(MD_hist, color='blue', bins=30)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
#axes[1, 1].legend()

ang_hist = abs(Angles_diff_no_GNL[mask_test == 1])
axes[1, 2].hist(ang_hist, color='blue', bins=30)
axes[1, 2].set_title('Angles')
axes[1, 2].set_xlabel('Angles [º]')
axes[1, 2].set_ylabel('Frequency')
#axes[1, 2].legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences no GNL correction [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
axes[0, 0].hist(S0_hist_thresholded, color='blue', bins=30)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
#axes[0, 0].legend()

d_par_hist = abs(AD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
axes[0, 1].hist(AD_hist_thresholded, color='blue', bins=30)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
#axes[0, 1].legend()

d_per_hist = abs(RD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
axes[0, 2].hist(RD_hist_thresholded, color='blue', bins=30)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
#axes[0, 2].legend()

FA_hist = abs(FA_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
axes[1, 0].hist(FA_hist_thresholded, color='blue', bins=30)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()

MD_hist = abs(MD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
axes[1, 1].hist(MD_hist_thresholded, color='blue', bins=30)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
#axes[1, 1].legend()

ang_hist = abs(np.nan_to_num(Angles_diff_no_GNL)[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
axes[1, 2].hist(ang_hist_thresholded, color='blue', bins=30)
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
#axes[1, 2].legend()

plt.tight_layout()
plt.show()


# scatter plots
pred_test_param_no_GNL_ = pred_test_param_no_GNL[:][mask_test == 1]

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
rgb_map_true_1D = np.zeros((height, 3), dtype=np.uint8)
rgb_map_true_1D[:, 0] = x_abs * 255   # Red channel
rgb_map_true_1D[:, 1] = y_abs * 255   # Green channel
rgb_map_true_1D[:, 2] = z_abs * 255   # Blue channel

# predicted angles no GNL correction
height = pred_test_params_no_GNL_vec_S.shape[0]

theta = pred_test_params_no_GNL_vec_ang1
phi = pred_test_params_no_GNL_vec_ang2

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred_no_GNL_1D = np.zeros((height, 3), dtype=np.uint8)
rgb_map_pred_no_GNL_1D[:, 0] = x_abs * 255   # Red channel
rgb_map_pred_no_GNL_1D[:, 1] = y_abs * 255   # Green channel
rgb_map_pred_no_GNL_1D[:, 2] = z_abs * 255   # Blue channel

dot_product = np.empty((rgb_map_true_1D.shape[0]))
for x in range(0, rgb_map_true_1D.shape[0], 1):
    vector1 = np.array(rgb_map_true_1D[x, :])
    vector1 = np.double(vector1)/np.linalg.norm(vector1)
    vector2 = np.array(rgb_map_pred_no_GNL_1D[x, :])
    vector2 = np.double(vector2)/np.linalg.norm(vector2)
    dot_product[x] = round(np.dot(vector1, vector2), 2)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.suptitle(f'Scatter plot GT vs LS prediction no GNL correction. [{num}]. idx: {idx_image}', fontsize=25)

axes[0].hist(1 - abs(dot_product), color='b', bins = 30)
axes[0].set_xlabel('1 - |a . b|')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Angles')

axes[1].scatter(true_test_params_vec_AD, pred_test_param_no_GNL_[:, 2], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_AD,  pred_test_param_no_GNL_[:, 2])
trend_line = slope * np.array(true_test_params_vec_AD) + intercept
axes[1].plot(true_test_params_vec_AD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[1].set_xlabel('True AD')
axes[1].set_ylabel('Predicted AD')
axes[1].set_title('AD')

axes[2].scatter(true_test_params_vec_RD, pred_test_param_no_GNL_[:, 3], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_RD,  pred_test_param_no_GNL_[:, 3])
trend_line = slope * np.array(true_test_params_vec_RD) + intercept
axes[2].plot(true_test_params_vec_RD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[2].set_xlabel('True RD')
axes[2].set_ylabel('Predicted RD')
axes[2].set_title('RD')

axes[3].scatter(true_test_params_vec_S, pred_test_param_no_GNL_[:, 4], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_S,  pred_test_param_no_GNL_[:, 4])
trend_line = slope * np.array(true_test_params_vec_S) + intercept
axes[3].plot(true_test_params_vec_S, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[3].set_xlabel('True S0')
axes[3].set_ylabel('Predicted S0')
axes[3].set_title('S0')

plt.tight_layout()
plt.show()


###################### GNL corretion #############################


# Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
axial_diffusivity = pred_test_param_GNL[:, :, 2]
radial_diffusivity = pred_test_param_GNL[:, :, 3]

# Define the diffusion tensor using the voxel-wise diffusivity data
diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
diffusion_tensor[..., 0] = radial_diffusivity
diffusion_tensor[..., 1] = radial_diffusivity
diffusion_tensor[..., 2] = axial_diffusivity

FA_pred_GNL = fractional_anisotropy(diffusion_tensor)
MD_pred_GNL = (pred_test_param_GNL[:, :, 2] + pred_test_param_GNL[:, :, 3] + pred_test_param_GNL[:, :, 3]) / 3

# predicted angles
height = pred_test_param_GNL.shape[0]
width = pred_test_param_GNL.shape[1]

theta = pred_test_param_GNL[:,:,0]
phi = pred_test_param_GNL[:,:,1]

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred_GNL = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred_GNL[:, :, 0] = x_abs * 255   # Red channel
rgb_map_pred_GNL[:, :, 1] = y_abs * 255   # Green channel
rgb_map_pred_GNL[:, :, 2] = z_abs * 255   # Blue channel

rgb_map_pred_GNL[:, :, 0] = np.where(mask_test == 1, rgb_map_pred_GNL[:, :, 0], np.nan)
rgb_map_pred_GNL[:, :, 1] = np.where(mask_test == 1, rgb_map_pred_GNL[:, :, 1], np.nan)
rgb_map_pred_GNL[:, :, 2] = np.where(mask_test == 1, rgb_map_pred_GNL[:, :, 2], np.nan)

rgb_map_pred2_GNL = np.zeros((height, width, 3), dtype=np.uint8)
rgb_map_pred2_GNL[:, :, 0] = rgb_map_pred_GNL[:, :, 0] * FA_pred_GNL   # Red channel
rgb_map_pred2_GNL[:, :, 1] = rgb_map_pred_GNL[:, :, 1] * FA_pred_GNL   # Green channel
rgb_map_pred2_GNL[:, :, 2] = rgb_map_pred_GNL[:, :, 2] * FA_pred_GNL   # Blue channel

fig, axes = plt.subplots(1, 2, figsize=(16, 8) , facecolor='black')
fig.suptitle('Angles with GNL correction', color='white')
im1 = axes[0].imshow(np.rot90(rgb_map_pred_GNL), cmap='jet')
axes[0].set_title('rgb_map', color='white', fontsize=20)
axes[0].axis('off')
# =============================================================================
# colorbar1 = plt.colorbar(im1, ax=axes[0])
# colorbar1.ax.yaxis.set_tick_params(color='white',  labelsize=20)
# =============================================================================
im2 = axes[1].imshow(np.rot90(rgb_map_pred2_GNL), cmap='jet')
axes[1].set_title('rgb_map * FA map', color='white', fontsize=20)
axes[1].axis('off')
# =============================================================================
# colorbar2 = plt.colorbar(im2, ax=axes[1])
# colorbar2.ax.yaxis.set_tick_params(color='white',  labelsize=15)
# =============================================================================
fig.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Results from fitting with LS WITH GNL corretion. Slice {[num]}', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 4]), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 2]), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 3]), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred_GNL), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred_GNL), cmap='gray')
plt.title('MD', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_pred2_GNL), cmap='jet')
plt.title('rgb_map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()

rgb_map_diff = np.empty((rgb_map_pred2_GNL.shape[0], rgb_map_pred2_GNL.shape[1]))
rgb_map_diff3 = np.empty((rgb_map_true2.shape[0], rgb_map_true2.shape[1]))

dot_product = np.empty((rgb_map_pred2_GNL.shape[0], rgb_map_pred2_GNL.shape[1]))
for pix_x in range(0, rgb_map_pred2_GNL.shape[0], 1):
    for pix_y in range(0, rgb_map_pred2_GNL.shape[1], 1):
        vector1 = np.array(rgb_map_pred2_GNL[pix_x, pix_y, :])
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
rgb_map_diff_GNL = np.where(mask_test == 1, rgb_map_diff, np.nan)
cmap = plt.get_cmap("jet")
cmap.set_bad("black", alpha=1.0)  # Map NaN values to black

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff_GNL), cmap=cmap)
plt.colorbar()
plt.title(f'Angle Differences WITH GNL correction')
plt.axis('off')
plt.show()

rgb_map_diff2_GNL = 1 - rgb_map_diff3

# Plot the angle differences in radians
plt.imshow(np.rot90(rgb_map_diff2_GNL), cmap=cmap)
plt.colorbar()
plt.title(f'Angle Differences WITH GNL correction. 1-diff')
plt.axis('off')
plt.show()


plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle('Plot difference (GT - LS) WITH GNL correction', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred_GNL - np.nan_to_num(FA_GT)), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred_GNL - np.nan_to_num(MD_GT)), cmap='gray')
plt.title('MD diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_diff2_GNL), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle('Plot difference (GT - LS) WITH GNL correction', fontsize=25, color='white')

plt.subplot(2, 3, 1)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)), cmap='gray')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)), cmap='gray')
plt.title('AD/d_par', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
plt.imshow(np.rot90(pred_test_param_GNL[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)), cmap='gray')
plt.title('RD/d_per', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
plt.imshow(np.rot90(FA_pred_GNL - np.nan_to_num(FA_GT)), cmap='gray')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
plt.imshow(np.rot90(MD_pred_GNL - np.nan_to_num(MD_GT)), cmap='gray')
plt.title('MD diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
plt.imshow(np.rot90(rgb_map_diff2_GNL), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


S0_diff_GNL = pred_test_param_GNL[:, :, 4] - np.nan_to_num(true_test_params_unvec[:, :, 4] * mask_test)
AD_diff_GNL = pred_test_param_GNL[:, :, 2] - np.nan_to_num(true_test_params_unvec[:, :, 2] * mask_test)
RD_diff_GNL = pred_test_param_GNL[:, :, 3] - np.nan_to_num(true_test_params_unvec[:, :, 3] * mask_test)
FA_diff_GNL = FA_pred_GNL - np.nan_to_num(FA_GT)
MD_diff_GNL = MD_pred_GNL - np.nan_to_num(MD_GT)
Angles_diff_GNL = rgb_map_diff2_GNL

plt.figure(figsize=(14, 9), facecolor='black')
plt.suptitle(f'Plot difference (GT - LS) no GNL correction. percentile{per}', fontsize=25, color='white')

plt.subplot(2, 3, 1)
S0_diff_flat = np.nan_to_num(S0_diff_GNL.flatten())
threshold_value = np.percentile(S0_diff_flat, per)
S0_diff_thresholded_GNL = np.clip(S0_diff_GNL, 0, threshold_value)
plt.imshow(np.rot90(S0_diff_thresholded_GNL), cmap='jet')
plt.title('S0', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 2)
AD_diff_flat = np.nan_to_num(AD_diff_GNL.flatten())
threshold_value = np.percentile(AD_diff_flat, per)
AD_diff_thresholded_GNL = np.clip(AD_diff_GNL, 0, threshold_value)
plt.imshow(np.rot90(AD_diff_thresholded_GNL), cmap='jet')
plt.title('AD/d_par [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 3)
RD_diff_flat = np.nan_to_num(RD_diff_GNL.flatten())
threshold_value = np.percentile(RD_diff_flat, per)
RD_diff_thresholded_GNL = np.clip(RD_diff_GNL, 0, threshold_value)
plt.imshow(np.rot90(RD_diff_thresholded_GNL), cmap='jet')
plt.title('RD/d_per [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 4)
FA_diff_flat = np.nan_to_num(FA_diff_GNL.flatten())
threshold_value = np.percentile(FA_diff_flat, per)
FA_diff_thresholded_GNL = np.clip(FA_diff_GNL, 0, threshold_value)
plt.imshow(np.rot90(FA_diff_thresholded_GNL), cmap='jet')
plt.title('FA map', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 5)
MD_diff_flat = np.nan_to_num(MD_diff_GNL.flatten())
threshold_value = np.percentile(MD_diff_flat, per)
MD_diff_thresholded_GNL = np.clip(MD_diff_GNL, 0, threshold_value)
plt.imshow(np.rot90(MD_diff_thresholded_GNL), cmap='jet')
plt.title('MD diff [μm^2/ms]', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.subplot(2, 3, 6)
ang_diff_flat = np.nan_to_num(Angles_diff_GNL.flatten())
threshold_value = np.percentile(ang_diff_flat, per)
ang_diff_thresholded_GNL = np.clip(Angles_diff_GNL, 0, threshold_value)
plt.imshow(np.rot90(ang_diff_thresholded_GNL), cmap='jet')
plt.title('rgb_map diff', color='white', fontsize=15)
plt.axis('off')
color_bar = plt.colorbar()
cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')
plt.setp(cbytick_obj, color='white')

plt.show()


# plot histograms 
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences WITH GNL correction [{num}]. idx: {idx_image}', fontsize=25)

S0_hist = abs(S0_diff_GNL[mask_test == 1])
axes[0, 0].hist(S0_hist, color='blue', bins=30)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
#axes[0, 0].legend()

d_par_hist = abs(AD_diff_GNL[mask_test == 1])
axes[0, 1].hist(d_par_hist, color='blue', bins=30)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_|| ')
axes[0, 1].set_ylabel('Frequency')
#axes[0, 1].legend()

d_per_hist = abs(RD_diff_GNL[mask_test == 1])
axes[0, 2].hist(d_per_hist, color='blue', bins=30)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
#axes[0, 2].legend()

FA_hist = abs(FA_diff_GNL[mask_test == 1])
axes[1, 0].hist(FA_hist, color='blue', bins=30)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()

MD_hist = abs(MD_diff_GNL[mask_test == 1])
axes[1, 1].hist(MD_hist, color='blue', bins=30)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
#axes[1, 1].legend()

ang2_nc_hist = abs(Angles_diff_GNL[mask_test == 1])
axes[1, 2].hist(ang2_nc_hist, color='blue', bins=30)
axes[1, 2].set_title('Angles')
axes[1, 2].set_xlabel('Angles [º]')
axes[1, 2].set_ylabel('Frequency')
#axes[1, 2].legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences no GNL correction [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff_GNL[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
axes[0, 0].hist(S0_hist_thresholded, color='blue', bins=30)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
#axes[0, 0].legend()

d_par_hist = abs(AD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
axes[0, 1].hist(AD_hist_thresholded, color='blue', bins=30)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
#axes[0, 1].legend()

d_per_hist = abs(RD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
axes[0, 2].hist(RD_hist_thresholded, color='blue', bins=30)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
#axes[0, 2].legend()

FA_hist = abs(FA_diff_GNL[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
axes[1, 0].hist(FA_hist_thresholded, color='blue', bins=30)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
#axes[1, 0].legend()

MD_hist = abs(MD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
axes[1, 1].hist(MD_hist_thresholded, color='blue', bins=30)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
#axes[1, 1].legend()

ang_hist = abs(np.nan_to_num(Angles_diff_GNL)[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
axes[1, 2].hist(ang_hist_thresholded, color='blue', bins=30)
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
#axes[1, 2].legend()

plt.tight_layout()
plt.show()

# scatter plots
pred_test_param_GNL_ = pred_test_param_GNL[:][mask_test == 1]

# predicted angles no GNL correction
height = pred_test_params_GNL_vec_S.shape[0]

theta = pred_test_params_GNL_vec_ang1
phi = pred_test_params_GNL_vec_ang2

# From spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Take the absolute values of the Cartesian coordinates
x_abs = np.abs(x)
y_abs = np.abs(y)
z_abs = np.abs(z)

# Create an RGB map with red, blue, and green
rgb_map_pred_GNL_1D = np.zeros((height, 3), dtype=np.uint8)
rgb_map_pred_GNL_1D[:, 0] = x_abs * 255   # Red channel
rgb_map_pred_GNL_1D[:, 1] = y_abs * 255   # Green channel
rgb_map_pred_GNL_1D[:, 2] = z_abs * 255   # Blue channel

dot_product = np.empty((rgb_map_true_1D.shape[0]))
for x in range(0, rgb_map_true_1D.shape[0], 1):
    vector1 = np.array(rgb_map_true_1D[x, :])
    vector1 = np.double(vector1)/np.linalg.norm(vector1)
    vector2 = np.array(rgb_map_pred_GNL_1D[x, :])
    vector2 = np.double(vector2)/np.linalg.norm(vector2)
    dot_product[x] = round(np.dot(vector1, vector2), 2)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.suptitle(f'Scatter plot GT vs LS prediction WITH GNL correction. [{num}]. idx: {idx_image}', fontsize=25)

axes[0].hist(1 - abs(dot_product), color='b', bins = 30)
axes[0].set_xlabel('1 - |a . b|')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Angles')

axes[1].scatter(true_test_params_vec_AD, pred_test_param_GNL_[:, 2], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_AD,  pred_test_param_GNL_[:, 2])
trend_line = slope * np.array(true_test_params_vec_AD) + intercept
axes[1].plot(true_test_params_vec_AD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[1].set_xlabel('True AD')
axes[1].set_ylabel('Predicted AD')
axes[1].set_title('AD')

axes[2].scatter(true_test_params_vec_RD, pred_test_param_GNL_[:, 3], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_RD,  pred_test_param_GNL_[:, 3])
trend_line = slope * np.array(true_test_params_vec_RD) + intercept
axes[2].plot(true_test_params_vec_RD, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[2].set_xlabel('True RD')
axes[2].set_ylabel('Predicted RD')
axes[2].set_title('RD')

axes[3].scatter(true_test_params_vec_S, pred_test_param_GNL_[:, 4], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    true_test_params_vec_S,  pred_test_param_GNL_[:, 4])
trend_line = slope * np.array(true_test_params_vec_S) + intercept
axes[3].plot(true_test_params_vec_S, trend_line,
             color='red', label='Trend Line', linestyle='--')
axes[3].set_xlabel('True S0')
axes[3].set_ylabel('Predicted S0')
axes[3].set_title('S0')

plt.tight_layout()
plt.show()

#sns.regplot(x=true_test_params_vec_RD, y=pred_test_param_GNL_[:, 3], marker="x", color=".3", line_kws=dict(color="r"))

num_bin = 30

# plot histograms  together
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Plot hist for differences together. [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
S0_hist2 = abs(S0_diff_GNL[mask_test == 1])
threshold_value = np.percentile(S0_hist2, per)
S0_hist_thresholded2 = np.clip(S0_hist2, 0, threshold_value)
axes[0, 0].hist(S0_hist_thresholded, color='blue', label='no GNL', bins = num_bin)
axes[0, 0].hist(S0_hist_thresholded2, color='red', label='GNL', bins = num_bin)
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('S0')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

d_par_hist = abs(AD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
d_par_hist2 = abs(AD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(d_par_hist2, per)
AD_hist_thresholded2 = np.clip(d_par_hist2, 0, threshold_value)
axes[0, 1].hist(AD_hist_thresholded, color='blue',  label='no GNL', bins = num_bin)
axes[0, 1].hist(AD_hist_thresholded2, color='red',label='GNL', bins = num_bin)
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('d_||')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

d_per_hist = abs(RD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
d_per_hist2 = abs(RD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(d_per_hist2, per)
RD_hist_thresholded2 = np.clip(d_per_hist2, 0, threshold_value)
axes[0, 2].hist(RD_hist_thresholded, color='blue', label='no GNL', bins = num_bin)
axes[0, 2].hist(RD_hist_thresholded2, color='red', label='GNL', bins = num_bin)
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('d_⊥')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

FA_hist = abs(FA_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
FA_hist2 = abs(FA_diff_GNL[mask_test == 1])
threshold_value = np.percentile(FA_hist2, per)
FA_hist_thresholded2 = np.clip(FA_hist2, 0, threshold_value)
axes[1, 0].hist(FA_hist_thresholded, color='blue',  label='no GNL', bins = num_bin)
axes[1, 0].hist(FA_hist_thresholded2, color='red',  label='GNL', bins = num_bin)
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('FA')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

MD_hist = abs(MD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
MD_hist2 = abs(MD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(MD_hist2, per)
MD_hist_thresholded2 = np.clip(MD_hist2, 0, threshold_value)
axes[1, 1].hist(MD_hist_thresholded, color='blue', label='no GNL', bins = num_bin)
axes[1, 1].hist(MD_hist_thresholded2, color='red',  label='GNL', bins = num_bin)
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('MD')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

ang_hist = abs(np.nan_to_num(Angles_diff_no_GNL)[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
ang_hist2 = abs(np.nan_to_num(Angles_diff_GNL)[mask_test == 1])
threshold_value = np.percentile(ang_hist2, per)
ang_hist_thresholded2 = np.clip(ang_hist2, 0, threshold_value)
axes[1, 2].hist(ang_hist_thresholded, color='blue', label='no GNL', bins = num_bin)
axes[1, 2].hist(ang_hist_thresholded2, color='red', label='GNL', bins = num_bin)
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Angles')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()


plt.tight_layout()
plt.show()


# Create ECDF plots
plt.figure(figsize=(10, 6))
plt.plot(np.sort(MD_hist_thresholded), np.linspace(0, 1, len(MD_hist_thresholded), endpoint=False), label='Array 1')
plt.plot(np.sort(MD_hist_thresholded2), np.linspace(0, 1, len(MD_hist_thresholded2), endpoint=False),  linewidth=3, linestyle='dotted',  markersize=3, label='Array 2')
plt.legend()
plt.title('Empirical Cumulative Distribution Function of Differences')
plt.xlabel('Difference')
plt.ylabel('Cumulative Probability')
plt.show()


# plot histograms  together
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
fig.suptitle(f'Cumulative Distribution Function of Differences. [{num}]. idx: {idx_image}. {per}th per', fontsize=20)

S0_hist = abs(S0_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(S0_hist, per)
S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
S0_hist2 = abs(S0_diff_GNL[mask_test == 1])
threshold_value = np.percentile(S0_hist2, per)
S0_hist_thresholded2 = np.clip(S0_hist2, 0, threshold_value)
axes[0, 0].plot(np.sort(S0_hist_thresholded), np.linspace(0, 1, len(S0_hist_thresholded), endpoint=False), label='no GNL')
axes[0, 0].plot(np.sort(S0_hist_thresholded2), np.linspace(0, 1, len(S0_hist_thresholded2), endpoint=False), linewidth=3, linestyle='dotted', label='GNL')
axes[0, 0].set_title('S0')
axes[0, 0].set_xlabel('Difference')
axes[0, 0].set_ylabel('Cum Prob')
axes[0, 0].legend()

d_par_hist = abs(AD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(d_par_hist, per)
AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
d_par_hist2 = abs(AD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(d_par_hist2, per)
AD_hist_thresholded2 = np.clip(d_par_hist2, 0, threshold_value)
axes[0, 1].plot(np.sort(AD_hist_thresholded), np.linspace(0, 1, len(AD_hist_thresholded), endpoint=False), label='no GNL')
axes[0, 1].plot(np.sort(AD_hist_thresholded2), np.linspace(0, 1, len(AD_hist_thresholded2), endpoint=False), linewidth=3, linestyle='dotted',label='GNL')
axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
axes[0, 1].set_xlabel('Difference')
axes[0, 1].set_ylabel('Cum Prob')
axes[0, 1].legend()

d_per_hist = abs(RD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(d_per_hist, per)
RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
d_per_hist2 = abs(RD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(d_per_hist2, per)
RD_hist_thresholded2 = np.clip(d_per_hist2, 0, threshold_value)
axes[0, 2].plot(np.sort(RD_hist_thresholded), np.linspace(0, 1, len(RD_hist_thresholded), endpoint=False), label='no GNL')
axes[0, 2].plot(np.sort(RD_hist_thresholded2), np.linspace(0, 1, len(RD_hist_thresholded2), endpoint=False), linewidth=3, linestyle='dotted',label='GNL')
axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
axes[0, 2].set_xlabel('Difference')
axes[0, 2].set_ylabel('Cum Prob')
axes[0, 2].legend()

FA_hist = abs(FA_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(FA_hist, per)
FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
FA_hist2 = abs(FA_diff_GNL[mask_test == 1])
threshold_value = np.percentile(FA_hist2, per)
FA_hist_thresholded2 = np.clip(FA_hist2, 0, threshold_value)
axes[1, 0].plot(np.sort(FA_hist_thresholded), np.linspace(0, 1, len(FA_hist_thresholded), endpoint=False), label='no GNL')
axes[1, 0].plot(np.sort(FA_hist_thresholded2), np.linspace(0, 1, len(FA_hist_thresholded2), endpoint=False), linewidth=3, linestyle='dotted',label='GNL')
axes[1, 0].set_title('FA map')
axes[1, 0].set_xlabel('Difference')
axes[1, 0].set_ylabel('Cum Prob')
axes[1, 0].legend()

MD_hist = abs(MD_diff_no_GNL[mask_test == 1])
threshold_value = np.percentile(MD_hist, per)
MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
MD_hist2 = abs(MD_diff_GNL[mask_test == 1])
threshold_value = np.percentile(MD_hist2, per)
MD_hist_thresholded2 = np.clip(MD_hist2, 0, threshold_value)
axes[1, 1].plot(np.sort(MD_hist_thresholded), np.linspace(0, 1, len(MD_hist_thresholded), endpoint=False), label='no GNL')
axes[1, 1].plot(np.sort(MD_hist_thresholded2), np.linspace(0, 1, len(MD_hist_thresholded2), endpoint=False), linewidth=3, linestyle='dotted',label='GNL')
axes[1, 1].set_title('MD [μm^2/ms]')
axes[1, 1].set_xlabel('Difference')
axes[1, 1].set_ylabel('Cum Prob')
axes[1, 1].legend()

ang_hist = abs(np.nan_to_num(Angles_diff_no_GNL)[mask_test == 1])
threshold_value = np.percentile(ang_hist, per)
ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
ang_hist2 = abs(np.nan_to_num(Angles_diff_GNL)[mask_test == 1])
threshold_value = np.percentile(ang_hist2, per)
ang_hist_thresholded2 = np.clip(ang_hist2, 0, threshold_value)
axes[1, 2].plot(np.sort(ang_hist_thresholded), np.linspace(0, 1, len(ang_hist_thresholded), endpoint=False),  label='no GNL')
axes[1, 2].plot(np.sort(ang_hist_thresholded2), np.linspace(0, 1, len(ang_hist_thresholded2), endpoint=False), linewidth=3, linestyle='dotted',label='GNL')
axes[1, 2].set_title('Angles [º]')
axes[1, 2].set_xlabel('Difference')
axes[1, 2].set_ylabel('Cum Prob')
axes[1, 2].legend()


plt.tight_layout()
plt.show()


## genereate boxplots

pairs_of_arrays = []
array1 = S0_hist_thresholded
array2 = S0_hist_thresholded2
pairs_of_arrays.append((array1, array2))

array1 = AD_hist_thresholded
array2 = AD_hist_thresholded2
pairs_of_arrays.append((array1, array2))

array1 = RD_hist_thresholded
array2 = RD_hist_thresholded2
pairs_of_arrays.append((array1, array2))

array1 = FA_hist_thresholded
array2 = FA_hist_thresholded2
pairs_of_arrays.append((array1, array2))

array1 = MD_hist_thresholded
array2 = MD_hist_thresholded2
pairs_of_arrays.append((array1, array2))

array1 = ang_hist_thresholded
array2 = ang_hist_thresholded2
pairs_of_arrays.append((array1, array2))

# Create a figure with 6 sets of boxplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

# Flatten the 2D array of subplots for easier indexing
axes = axes.flatten()

params = ['S0', 'AD', 'RD', 'FA', 'MD', 'Ang']                   

# Create a figure with 6 sets of boxplots
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
plt.suptitle(f'Plot differences with GT of LS pred WITHOUT and WITH GNL correction. [{num}]. idx: {idx_image}. {per}th per', fontsize=15)
axes = axes.flatten()
a = 0
for i, (array1, array2) in enumerate(pairs_of_arrays):
    sns.set_style("whitegrid")
    sns.boxplot(data=[array1, array2], ax=axes[i], width=0.4, palette='dark', showfliers=False)
    axes[i].set_title(f'{params[a]}')
    axes[i].set_xticklabels(['no GNL', 'GNL'])
    a = a+1
# Adjust layout and show the plot
plt.tight_layout()
plt.show()











