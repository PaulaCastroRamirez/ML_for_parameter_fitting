# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:09:10 2023

@author: pcastror

Visualize results SIMULATIONS from LS fitting

"""

from scipy.stats import circmean, circvar
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from scipy import stats
from dipy.io.image import load_nifti

path_pred = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_simulation_params.nii'
path_true = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\Simulations\exp1_MRImodelZeppelin_ntrain8000_nval2000_ntest1000_SNR70\syn_GT'


################## GROUND TRUTH PARAMETERS #########################

with open('{}_sigtest.bin'.format(path_true), 'rb') as file:
    file_truetest = os.path.basename(file.name)
    true_test_signal = pk.load(file)
    

with open('{}_paramtest.bin'.format(path_true), 'rb') as file:
    file_truetest = os.path.basename(file.name)
    true_test_param = pk.load(file)
   
true_test_param[:,3] = true_test_param[:,2] * true_test_param[:,3]


with open('{}_sigval.bin'.format(path_true), 'rb') as file:
    file_truetest = os.path.basename(file.name)
    true_val_signal = pk.load(file)
    

with open('{}_paramval.bin'.format(path_true), 'rb') as file:
    file_truetest = os.path.basename(file.name)
    true_val_param = pk.load(file)
    
true_val_param[:,3] = true_val_param[:,2] * true_val_param[:,3]


################## PREDICTED PARAMETERS #########################

loss_tissuetest, affine = load_nifti(path_pred)
loss_tissuetest[:, 3] = loss_tissuetest[:, 2] * loss_tissuetest[:,3]  # calculate d_per from k_per
param_name = ['Ang1', 'Ang2', 'AD', 'RD', 'S0']                   

print('PARAMETERS: ', param_name)

# Verify the size of the bin file
print('Shape of the loss_tissuetest (predicted parameters): ', loss_tissuetest.shape)
print('Number of samples loss_tissuetest: ', loss_tissuetest.shape[0])

parameters_tissuetest = loss_tissuetest.shape[1]


################## EVALUATE ANGLES #########################

print('')
print('EVALUATE ANGLES')

# true and predicted angles in radians ANG1
true_angles = true_test_param[:,0]
predicted_angles = loss_tissuetest[:,0]
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
true_angles = true_test_param[:,1]
predicted_angles = loss_tissuetest[:,0]
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
angles_true =  true_test_param[:,0]
angles_pred = loss_tissuetest[:,0]
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

angles_true =  true_test_param[:,1]
angles_pred = loss_tissuetest[:,1]
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

#### correct dot product
## true angles 
# Define the size of your image
height = true_test_param[:,0].shape[0]

theta = true_test_param[:,0] 
phi = true_test_param[:,1]

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

#print(rgb_map_true.shape)

## predicted angles
# Define the size of your image
height = loss_tissuetest[:,0].shape[0]

theta = loss_tissuetest[:,0] 
phi = loss_tissuetest[:,1]

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


#print(rgb_map_pred.shape)

dot_product = np.empty((rgb_map_true.shape[0]))
for x in range(0,rgb_map_true.shape[0],1):
    vector1 = np.array(rgb_map_true[x,:])
    vector1 = np.double(vector1)/np.linalg.norm(vector1)
    vector2 = np.array(rgb_map_pred[x,:])
    vector2 = np.double(vector2)/np.linalg.norm(vector2)
    dot_product[x] = round(np.dot(vector1, vector2),2)

plt.hist(1 - abs(dot_product), color='b')
plt.xlabel('1 - |a . b|')
plt.ylabel('Frequency')
plt.title('Histogram 1 - |a . b| angles, being a= true and b=pred')
plt.show()

print(1 - abs(dot_product))


print('EVALUATE ANGLES')

from scipy.stats import circmean, circvar
# Example true and predicted angles in radians
true_angles = true_test_param[:,0]
predicted_angles = loss_tissuetest[:,0]

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

# Example true and predicted angles in radians
true_angles = true_test_param[:,1]
predicted_angles = loss_tissuetest[:,1]

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



angles_true = true_test_param[:, 0]
angles_pred = loss_tissuetest[:, 0]
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


angles_true = true_test_param[:, 1]
angles_pred = loss_tissuetest[:, 1]
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



### FINAL PLOT :)
# Scatter plots of simulation results obtained on the test set. predicted signal (y-axis) against input measurements (x-axis)
fig, axes = plt.subplots(1, parameters_tissuetest-1, figsize=(15, 4))

axes[0].hist(1 - abs(dot_product), color='b')
axes[0].set_xlabel('1 - |a . b|')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Angles')

# Loop through each parameter and create a scatter plot
for param in range(2,parameters_tissuetest):
    axes[param-1].scatter(true_test_param[:, param], loss_tissuetest[:, param], s=5)
    # Fit a linear regression line to the data to plot trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_test_param[:, param],  loss_tissuetest[:, param])
    trend_line = slope * np.array(true_test_param[:, param]) + intercept
    axes[param-1].plot(true_test_param[:, param], trend_line, color='red', label='Trend Line', linestyle='--')    
    axes[param-1].set_xlabel(f'True {param_name[param]}')
    axes[param-1].set_ylabel(f'Predicted {param_name[param]}')
    axes[param-1].set_title(f'{param_name [param]}')
    
fig.suptitle('Scatter Plots of True vs ML prediction TEST', fontsize=16)

plt.tight_layout()  
plt.show()



   


