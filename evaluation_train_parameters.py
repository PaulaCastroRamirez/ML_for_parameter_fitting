# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:40:09 2023
@author: pcastror
Visualize training results from train_parameters

"""

import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.stats import circmean, circvar
from dipy.io.image import load_nifti
from dipy.reconst.dti import fractional_anisotropy
import torch

# =============================================================================
# path_pred = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\ML_strat1\results_Exp1_nhidden108-56-5_pdrop0.0_noepoch600_lr0.0001_mbatch100_seed12345'
# path_GT = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\Simulations\exp1_MRImodelZeppelin_ntrain8000_nval2000_ntest1000_SNR70\syn_GT'
# =============================================================================

path_pred = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\ML_strat1\results_prueba4_nhidden108-56-5_pdrop0.2_noepoch155_lr0.0005_mbatch100_seed12345'
path_GT = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\ML_strat1\samples\1000307_'
path_msk_test = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\ML_strat1\samples\1000307__mask_sigtest.bin'  # in case we are evaluating real data


idx_image = '100307'

path_true_params = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_' + idx_image + '_params.nii'
path_true_mask = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_' + idx_image
num = 72
with open('{}_mask.bin'.format(path_true_mask), 'rb') as file:
    # Load the object from the file
    file_loss_tissueval = os.path.basename(file.name)
    mask_test = pk.load(file)
mask_test = mask_test[:, :, num]

true_test_param2, affine = load_nifti(path_true_params)
true_test_params_unvec_1 = true_test_param2[:, :, num, :]                                          # extract a specific slice
true_test_params_unvec_1[:,:,3] = true_test_params_unvec_1[:,:,2]*true_test_params_unvec_1[:,:,3]      # calculate d_per from k_per

plt.imshow(np.rot90(true_test_params_unvec_1[:,:,3]), cmap='gray')
plt.show()

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


recon = 1  ## set this is you are evaluating simulations to 0, and real data to 1

print('') 
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('Device used. using GPU (cuda) will significantly speed up the training.', device) #Device used: using GPU will significantly speed up the training.


with open('{}_sigtest.bin'.format(path_GT), 'rb') as file:
    true_test_signal = pk.load(file)
    
with open('{}_paramtest.bin'.format(path_GT), 'rb') as file:
    true_test_param = pk.load(file)
true_test_param[:,3] = true_test_param[:,2] * true_test_param[:,3]

with open('{}_sigval.bin'.format(path_GT), 'rb') as file:
    true_val_signal = pk.load(file)
    
    
with open('{}_paramval.bin'.format(path_GT), 'rb') as file:
    true_val_param = pk.load(file)
true_val_param[:,3] = true_val_param[:,2] * true_val_param[:,3]

# =============================================================================
# data = true_test_param
# 
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# 
# # Step 1: Standardize the Data
# data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
# 
# # Step 2: Apply PCA
# pca = PCA(n_components=2)  # You can change n_components as needed
# data_pca = pca.fit_transform(data_standardized)
# 
# # Step 3: Perform Clustering
# kmeans = KMeans(n_clusters=100)  # You can change n_clusters as needed
# kmeans.fit(data_pca)
# cluster_labels = kmeans.labels_
# 
# # Step 4: Visualize the Clusters
# plt.figure(figsize=(8, 6))
# plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='viridis', s=10)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red', label='Centroids')
# plt.title('PCA Scatter Plot with Clusters')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.grid(True)
# plt.show()
# 
# =============================================================================


### losses
# =============================================================================
# with open( '{}\epoch600_net.pth'.format(path_pred), 'rb') as file:
#     model = pk.load(file)
# print('Model at epoch 600: ', model)
# =============================================================================
    
with open('{}\losstrain.bin'.format(path_pred), 'rb') as file:
    loss_train = pk.load(file)

with open('{}\lossval.bin'.format(path_pred), 'rb') as file:
    loss_val = pk.load(file)

print('Size of the training loss: ', loss_train.shape)
print('Size of the validation loss: ', loss_val.shape)
print('Number of epoch: ', loss_train.shape[0])
print('Number of mini-batches in train loss: ', loss_train.shape[1])
print('Number of mini-batches in validation loss: ', loss_val.shape[1])

epochs = loss_train.shape[0]

new_loss_train = np.zeros((loss_train.shape[0],1))
new_loss_val = np.zeros((loss_val.shape[0],1))

number_minibatches_train = np.array(loss_train.shape[1], dtype=np.float64)
number_minibatches_val = np.array(loss_val.shape[1], dtype=np.float64)

for i in range(0,epochs):
    sum_row_train = sum(loss_train[i,:])
    sum_row_val = sum(loss_val[i,:])
    new_loss_train[i,0] = sum_row_train/number_minibatches_train
    new_loss_val[i,0] = sum_row_val/number_minibatches_val


fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)
epochs = range(1, loss_train.shape[0]+1)
axes[0].plot(epochs, new_loss_train)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('loss_train')
axes[0].grid(True)
axes[1].plot(epochs, new_loss_val)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('loss_val')
axes[1].grid(True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)
axes[0].plot(epochs, new_loss_train)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('losstrain', fontsize=20)
axes[0].grid(True)
axes[1].plot(epochs, new_loss_val)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('lossval' , fontsize=20)
axes[1].grid(True)
plt.show()

plt.plot(epochs, new_loss_train, label='losstrain')
plt.plot(epochs, new_loss_val, label='lossval')
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Losses train vs validation')
plt.legend()
plt.show()
print('                               ')


################## signals ###############
with open('{}\lossvalmin_sigval.bin'.format(path_pred), 'rb') as file:
    loss_sigval = pk.load(file)

with open('{}\lossvalmin_sigtest.bin'.format(path_pred), 'rb') as file:
    loss_sigtest = pk.load(file)

print('Size of the loss_sigval loss: ', loss_sigval.shape)
print('Size of the loss_sigtest loss: ', loss_sigtest.shape)
print('Number of samples loss_sigval: ', loss_sigval.shape[0])
print('Number of samples loss_sigtest: ', loss_sigtest.shape[0])
print('Number of mini-batches in loss_sigval loss: ', loss_sigval.shape[1])
print('Number of mini-batches in loss_sigtest loss: ', loss_sigtest.shape[1])

samples_sigval = loss_sigval.shape[0]
samples_sigtest = loss_sigtest.shape[0]

num = 10
# plot signals
plt.plot(loss_sigval[num,:])
plt.tight_layout()
plt.title(f'MRI SIGNALS sigval [{num}]')
plt.show()
plt.plot(loss_sigtest[num,:])
plt.tight_layout()
plt.title(f'MRI SIGNALS sigtest [{num}]')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)
axes[0].plot(loss_sigtest[num,:])
axes[0].set_xlabel('measurements')
axes[0].set_ylabel('signal')
axes[0].set_title(f'pred_sig_test [{num}]', fontsize=20)
axes[0].grid(True)
axes[1].plot(true_test_signal[num,:])
axes[1].set_xlabel('measurements')
axes[1].set_ylabel('signal')
axes[1].set_title(f'true_sig_test [{num}]', fontsize=20)
axes[1].grid(True)
plt.show()

plt.plot(loss_sigtest[num,:], label='predicted')
plt.plot(true_test_signal[num,:], label='true')
plt.xlabel('value')
plt.ylabel('signal')
plt.title(f'pred_sig_test [{num}] vs true_sig_test [{num}]' )
plt.legend()
plt.show()

print('                               ')

################## parameters. scatter plots ###############

with open('{}\lossvalmin_tissueval.bin'.format(path_pred), 'rb') as file:
    loss_tissueval = pk.load(file)
loss_tissueval[:,3] = loss_tissueval[:,2] * loss_tissueval[:,3]

with open('{}\lossvalmin_tissuetest.bin'.format(path_pred), 'rb') as file:
    loss_tissuetest = pk.load(file)
loss_tissuetest[:,3] = loss_tissuetest[:,2] * loss_tissuetest[:,3]

param_name = ['ang1','ang2','AD','RD', 'S0']

print('Size of the loss_tissueval loss: ', loss_tissueval.shape)
print('Size of the loss_tissuetest loss: ', loss_tissuetest.shape)
print('Number of samples loss_tissueval: ', loss_tissueval.shape[0])
print('Number of samples loss_tissuetest: ', loss_tissuetest.shape[0])
print('Number of mini-batches in loss_tissueval loss: ', loss_tissueval.shape[1])
print('Number of mini-batches in loss_tissuetest loss: ', loss_tissuetest.shape[1])

parameters_tissueval = loss_tissueval.shape[1]
parameters_tissuetest = loss_tissuetest.shape[1]


# Scatter plots test set. predicted signal (y-axis) against input measurements (x-axis)
fig, axes = plt.subplots(1, parameters_tissuetest, figsize=(15, 4))
for param in range(parameters_tissuetest):
    axes[param].scatter(true_test_param[:, param], loss_tissuetest[:, param],s=5)
    axes[param].set_xlabel(f'True {param_name[param]}')
    axes[param].set_ylabel(f'Predicted {param_name[param]}')
    axes[param].set_title(f'{param_name [param]}')    
fig.suptitle('Scatter Plots of True vs ML prediction TEST', fontsize=16)
plt.tight_layout() 
plt.show()


RD_max = np.max(true_test_param[:, 3])
print('RD max GT: ', RD_max)
AD_max = np.max(true_test_param[:, 2])
print('AD max GT: ', AD_max)
print('')
AD_max = np.max(loss_tissuetest[:,2])
print('AD max pred: ', AD_max)
RD_max = np.max(loss_tissuetest[:,3])
print('RD max pred: ', RD_max)
print('')

pred_test_params_vec_S    = loss_tissuetest[:,4]
pred_test_params_vec_AD   = loss_tissuetest[:,2]
pred_test_params_vec_RD   = loss_tissuetest[:,3]
pred_test_params_vec_ang1 = loss_tissuetest[:,0]
pred_test_params_vec_ang2 = loss_tissuetest[:,1]

true_test_params_vec_S    = true_test_param[:,4]
true_test_params_vec_AD   = true_test_param[:,2]
true_test_params_vec_RD   = true_test_param[:,3]
true_test_params_vec_ang1 = true_test_param[:,0]
true_test_params_vec_ang2 = true_test_param[:,1]

parameters_tissuetest = loss_tissuetest.shape[1]

#### dot product ####
########### GT rgb map  ################
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

###########  predicted rgb map ############
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

## Calculate dot product between both rgb maps to plot hitogram of differences between the angles
dot_product = np.empty((rgb_map_true.shape[0]))
for x in range(0,rgb_map_true.shape[0],1):
    vector1 = np.array(rgb_map_true[x,:])
    vector1 = np.double(vector1)/np.linalg.norm(vector1)
    vector2 = np.array(rgb_map_pred[x,:])
    vector2 = np.double(vector2)/np.linalg.norm(vector2)
    dot_product[x] = round(np.dot(vector1, vector2),2)
    
print('')


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.suptitle('Scatter plot GT vs ML prediction test set', fontsize=20)
axes[0].hist(1 - abs(dot_product), color='b')
axes[0].set_xlabel('1 - |a . b|')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Angles')
axes[1].scatter(true_test_params_vec_AD, loss_tissuetest[:, 2], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(true_test_params_vec_AD,  loss_tissuetest[:, 2])
trend_line = slope * np.array(true_test_params_vec_AD) + intercept
axes[1].plot(true_test_params_vec_AD, trend_line, color='red', label='Trend Line', linestyle='--')    
axes[1].set_xlabel('True AD')
axes[1].set_ylabel('Predicted AD')
axes[1].set_title('AD')
axes[2].scatter(true_test_params_vec_RD, loss_tissuetest[:, 3], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(true_test_params_vec_RD,  loss_tissuetest[:, 3])
trend_line = slope * np.array(true_test_params_vec_RD) + intercept
axes[2].plot(true_test_params_vec_RD, trend_line, color='red', label='Trend Line', linestyle='--')    
axes[2].set_xlabel('True RD')
axes[2].set_ylabel('Predicted RD')
axes[2].set_title('RD')
axes[3].scatter(true_test_params_vec_S, loss_tissuetest[:, 4], s=5)
# Fit a linear regression line to the data to plot trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(true_test_params_vec_S,  loss_tissuetest[:, 4])
trend_line = slope * np.array(true_test_params_vec_S) + intercept
axes[3].plot(true_test_params_vec_S, trend_line, color='red', label='Trend Line', linestyle='--')    
axes[3].set_xlabel('True S0')
axes[3].set_ylabel('Predicted S0')
axes[3].set_title('S0')
plt.tight_layout()
plt.show()

print('')
print('')

# Calculate statistics
S0_mean = np.mean(pred_test_params_vec_S)
S0_min = np.min(pred_test_params_vec_S)
S0_max = np.max(pred_test_params_vec_S)

dpar_mean = np.mean(pred_test_params_vec_AD)
dpar_min = np.min(pred_test_params_vec_AD)
dpar_max = np.max(pred_test_params_vec_AD)

dper_mean = np.mean(pred_test_params_vec_RD)
dper_min = np.min(pred_test_params_vec_RD)
dper_max = np.max(pred_test_params_vec_RD)

ang2_mean = np.mean(pred_test_params_vec_ang1)
ang2_min = np.min(pred_test_params_vec_ang1)
ang2_max = np.max(pred_test_params_vec_ang1)

ang1_mean = np.mean(pred_test_params_vec_ang2)
ang1_min = np.min(pred_test_params_vec_ang2)
ang1_max = np.max(pred_test_params_vec_ang2)

# Create a table to display the statistics
statistics_table = {
    'noGNL ML': ['S0', 'dper', 'dpar', 'ang1', 'ang2'],
    'Mean': [S0_mean, dper_mean, dpar_mean, ang1_mean, ang2_mean],
    'Min': [S0_min, dper_min, dpar_min, ang1_min, ang2_min],
    'Max': [S0_max, dper_max, dpar_max, ang1_max, ang2_max]
}


df_stats = pd.DataFrame(statistics_table)
print(df_stats)

# Calculate statistics
S0_mean = np.mean(true_test_params_vec_S)
S0_min = np.min(true_test_params_vec_S)
S0_max = np.max(true_test_params_vec_S)

dpar_mean = np.mean(true_test_params_vec_AD)
dpar_min = np.min(true_test_params_vec_AD)
dpar_max = np.max(true_test_params_vec_AD)

dper_mean = np.mean(true_test_params_vec_RD)
dper_min = np.min(true_test_params_vec_RD)
dper_max = np.max(true_test_params_vec_RD)

ang2_mean = np.mean(true_test_params_vec_ang1)
ang2_min = np.min(true_test_params_vec_ang1)
ang2_max = np.max(true_test_params_vec_ang1)

ang1_mean = np.mean(true_test_params_vec_ang2)
ang1_min = np.min(true_test_params_vec_ang2)
ang1_max = np.max(true_test_params_vec_ang2)

print('')

# Create a table to display the statistics
statistics_table = {
    'trueGT': ['S0', 'dper', 'dpar', 'ang1', 'ang2'],
    'Mean': [S0_mean, dper_mean, dpar_mean, ang1_mean, ang2_mean],
    'Min': [S0_min, dper_min, dpar_min, ang1_min, ang2_min],
    'Max': [S0_max, dper_max, dpar_max, ang1_max, ang2_max]
}


df_stats = pd.DataFrame(statistics_table)
print(df_stats)



################## lossvalmin_net. ADDITIONAL INFORMATION ###############

# =============================================================================
# with open('{}\lossvalmin_net.bin'.format(path_pred), 'rb') as file:
#     lossvalmin_net = pk.load(file)
#     
# print('Model at best epoch: ', lossvalmin_net)
# print('')
# =============================================================================

file_path_best_epoch = path_pred + '\lossvalmin.info'
file_path_lowest_lossval = path_pred + '\lossval_min.txt'

with open(file_path_best_epoch, "r") as file:
    best_epoch = file.read()
    # Process the content of the .info file as needed
    print('The best epoch: ', best_epoch)
    
with open(file_path_lowest_lossval, "r") as file:
    lowest_lossval = file.read()
    # Process the content of the .info file as needed
    print('The lowest val loss: ', lowest_lossval)


## try plot catesian coordinates instead of spherical

ang1_nc = true_test_param[:,0]
ang2_nc = true_test_param[:,1]
height = ang1_nc.shape
width = ang1_nc.shape
phi = ang1_nc
theta = ang2_nc 
# From spherical coordinates to Cartesian coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
# Take the absolute values of the Cartesian coordinates
x_abs_true = np.abs(x)
y_abs_true = np.abs(y)
z_abs_true = np.abs(z)

ang1_nc = loss_tissuetest[:,0]
ang2_nc = loss_tissuetest[:,1]
height = ang1_nc.shape
width = ang1_nc.shape
phi = ang1_nc
theta = ang2_nc 
# From spherical coordinates to Cartesian coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
# Take the absolute values of the Cartesian coordinates
x_abs_pred = np.abs(x)
y_abs_pred = np.abs(y)
z_abs_pred = np.abs(z)


fig, axes = plt.subplots(1, 3, figsize=(30, 10))
fig.subplots_adjust(hspace=0.5)
axes[0].scatter(x_abs_true,x_abs_pred)
axes[0].set_xlabel('True', fontsize=20)
axes[0].set_ylabel('Pred', fontsize=20)
axes[0].set_title('Angle x (º)', fontsize=30)
axes[0].grid(True)
axes[1].scatter(y_abs_true,y_abs_pred)
axes[1].set_xlabel('True', fontsize=20)
axes[1].set_ylabel('Pred', fontsize=20)
axes[1].set_title('Angle y (º)', fontsize=30)
axes[1].grid(True)
axes[2].scatter(z_abs_true,z_abs_pred)
axes[2].set_xlabel('True', fontsize=20)
axes[2].set_ylabel('Pred', fontsize=20)
axes[2].set_title('Angle z (º)', fontsize=30)
axes[2].grid(True)


fig, axes = plt.subplots(1, 2, figsize=(30, 10))
fig.subplots_adjust(hspace=0.5)
axes[0].scatter(true_test_param[:,0],loss_tissuetest[:,0])
axes[0].set_xlabel('True', fontsize=20)
axes[0].set_ylabel('Pred', fontsize=20)
axes[0].set_title('Angle 1 (rad)', fontsize=30)
axes[0].grid(True)
axes[1].scatter(true_test_param[:,1],loss_tissuetest[:,1])
axes[1].set_xlabel('True', fontsize=20)
axes[1].set_ylabel('Pred', fontsize=20)
axes[1].set_title('Angle 2 (rad)', fontsize=30)
axes[1].grid(True)

print('')

# ang1
true_angles = true_test_param[:,0]
predicted_angles = loss_tissuetest[:,0]
mean_true = circmean(true_angles)
mean_predicted = circmean(predicted_angles)
variance_true = circvar(true_angles)
variance_predicted = circvar(predicted_angles)
print('')
print(f"True Circular Mean ang1: {mean_true} radians")
print(f"Predicted Circular Mean ang1: {mean_predicted} radians")
print(f"True Circular Variance ang1: {variance_true}")
print(f"Predicted Circular Variance ang1: {variance_predicted}")

# ang2
true_angles = true_test_param[:,1]
predicted_angles = loss_tissuetest[:,1]
mean_true = circmean(true_angles)
mean_predicted = circmean(predicted_angles)
variance_true = circvar(true_angles)
variance_predicted = circvar(predicted_angles)
print('')
print(f"True Circular Mean ang2: {mean_true} radians")
print(f"Predicted Circular Mean ang2: {mean_predicted} radians")
print(f"True Circular Variance ang2: {variance_true}")
print(f"Predicted Circular Variance ang2: {variance_predicted}")

# ang1
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

# ang2
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


## distributions
Parameters = ['ang(1)', 'ang(2)', 'd_z_par', 'd_z_per', 'S0']
fig, axs = plt.subplots(1, loss_tissuetest.shape[1], figsize=(15, 3))
fig.subplots_adjust(hspace=0.5)
fig.suptitle('output distributions (predictions)')
for i in range(0,loss_tissuetest.shape[1]):
    axs[i].hist(loss_tissuetest[:, i], bins=50, density=True, color='b')
    axs[i].set_xlabel('Value', fontsize = 10)
    axs[i].set_ylabel('Frequency', fontsize = 10)
    axs[i].set_title(f'{Parameters[i]}', fontsize = 10)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, true_test_param.shape[1], figsize=(15, 3))
fig.subplots_adjust(hspace=0.5)
fig.suptitle('input distributions (simulations)')
for i in range(0,true_test_param.shape[1]):
    axs[i].hist(true_test_param[:, i], bins=50, density=True, color='b')
    axs[i].set_xlabel('Value', fontsize = 10)
    axs[i].set_ylabel('Frequency', fontsize = 10)
    axs[i].set_title(f'{Parameters[i]}', fontsize = 10)
plt.tight_layout()
plt.show()

print(' ')


### see if the angles are off in the locations where dpar and dper are similar. in that case,
# the diffusiont tensor becomes isotropic
data = {'Pred Ang1': loss_tissuetest[:,0],
        'True Ang2': true_test_param[:,1],
        'Pred AD - Pred RD': np.abs(loss_tissuetest[:,2]- loss_tissuetest[:,3]),
        'True - Pred Ang1': np.abs(true_test_param[:,0] - loss_tissuetest[:,0]),
        'True - Pred Ang2': np.abs(true_test_param[:,1] - loss_tissuetest[:,2])}
df = pd.DataFrame(data)
print(df)

import numpy as np
from scipy.stats import circmean

print('')

## more info
euclidean_distance0 = np.linalg.norm(true_test_param[:,0] - loss_tissuetest[:,0])
euclidean_distance1 = np.linalg.norm(true_test_param[:,1] - loss_tissuetest[:,1])
euclidean_distance2 = np.linalg.norm(true_test_param[:,2] - loss_tissuetest[:,2])
euclidean_distance3 = np.linalg.norm(true_test_param[:,3] - loss_tissuetest[:,3])
euclidean_distance4 = np.linalg.norm(true_test_param[:,4] - loss_tissuetest[:,4])

print("Euclidean Distance angle 1:", euclidean_distance0)
print("Euclidean Distance angle 2:", euclidean_distance1)
print("Euclidean Distance AD:", euclidean_distance2)
print("Euclidean Distance RD:", euclidean_distance3)
print("Euclidean Distance S0:", euclidean_distance4)
print('')



if recon == 1:
	## reconstruct images and evaluate
	with open(path_msk_test, 'rb') as file:
	    mask_test = pk.load(file)
	
	indices = np.where(mask_test == 1)
	true_test_params_unvec = np.zeros((mask_test.shape[0],mask_test.shape[1], mask_test.shape[2], 5))
	true_test_params_unvec[indices] = true_test_param
	
	print(np.max(true_test_params_unvec[:,:,0,3]))
	
	
	indices = np.where(mask_test == 1)
	params_test = np.zeros((mask_test.shape[0],mask_test.shape[1], mask_test.shape[2], 5))
	params_test[indices] = loss_tissuetest
	
	print(params_test.shape)
	
	
	############### GT IMAGES ####################
	axial_diffusivity= true_test_params_unvec[:,:,:,2]
	radial_diffusivity = true_test_params_unvec[:,:,:,3]
	
	# Define the diffusion tensor using the voxel-wise diffusivity data
	diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], params_test.shape[2], 3))
	diffusion_tensor[..., 0] = radial_diffusivity
	diffusion_tensor[..., 1] = radial_diffusivity
	diffusion_tensor[..., 2] = axial_diffusivity
	
	FA_true = fractional_anisotropy(diffusion_tensor) * mask_test
	MD_true =  ((axial_diffusivity + radial_diffusivity + radial_diffusivity) / 3) * mask_test
	
	# rgb GT image
	height = true_test_params_unvec.shape[0]
	width = true_test_params_unvec.shape[1]
	
	theta = true_test_params_unvec[:,:,:,0]
	phi = true_test_params_unvec[:,:,:,1]
	
	# From spherical coordinates to Cartesian coordinates
	x = np.sin(theta) * np.cos(phi)
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(theta)
	
	# Take the absolute values of the Cartesian coordinates
	x_abs = np.abs(x)
	y_abs = np.abs(y)
	z_abs = np.abs(z)
	
	# Create an RGB map with red, blue, and green
	rgb_map_true = np.zeros((height, width, params_test.shape[2], 3), dtype=np.uint8)
	rgb_map_true[:, :, :, 0] = x_abs * 255   # Red channel
	rgb_map_true[:, :, :, 1] = y_abs * 255   # Green channel
	rgb_map_true[:, :, :, 2] = z_abs * 255   # Blue channel
	
	rgb_map_true[:,:,:,0] = np.where(mask_test == 1, rgb_map_true[:,:,:,0], np.nan)
	rgb_map_true[:,:,:,1] = np.where(mask_test == 1, rgb_map_true[:,:,:,1], np.nan)
	rgb_map_true[:,:,:,2] = np.where(mask_test == 1, rgb_map_true[:,:,:,2], np.nan)
	
	rgb_map_true2 = np.zeros((height, width, params_test.shape[2], 3), dtype=np.uint8)
	rgb_map_true2[:, :, :, 0] = rgb_map_true[:, :, :, 0] * FA_true   # Red channel
	rgb_map_true2[:, :, :, 1] = rgb_map_true[:, :, :, 1] * FA_true   # Green channel
	rgb_map_true2[:, :, :, 2] = rgb_map_true[:, :, :, 2] * FA_true   # Blue channel
	
	
	############### PREDICTED ML IMAGES ####################
	with open('{}/lossvalmin_tissuetest.bin'.format(path_pred), 'rb') as file:
	    loss_tissuetest = pk.load(file)
	
	print(loss_tissuetest.shape)
	
	loss_tissuetest[:,3] = loss_tissuetest[:,2] * loss_tissuetest[:,3]  
	param_name = ['Ang1','Ang2','AD','RD', 'S0']
	
	indices = np.where(mask_test == 1)
	restored_param = np.zeros((mask_test.shape[0],mask_test.shape[1], mask_test.shape[2], 5))
	restored_param[indices] = loss_tissuetest
	#print(restored_param.shape)
	
	AD_pred = restored_param [:,:,:,2]
	RD_pred = restored_param [:,:,:,3]
	S0_pred = restored_param [:,:,:,4]
	ang1_pred = restored_param [:,:,:,0] 
	ang2_pred = restored_param [:,:,:,1] 
	
	
	#### CALCULATE FA ####
	# # Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
	axial_diffusivity= AD_pred
	radial_diffusivity = RD_pred
	
	# Define the diffusion tensor using the voxel-wise diffusivity data
	diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], params_test.shape[2], 3))
	diffusion_tensor[..., 0] = radial_diffusivity
	diffusion_tensor[..., 1] = radial_diffusivity
	diffusion_tensor[..., 2] = axial_diffusivity
	
	FA_pred = fractional_anisotropy(diffusion_tensor)
	MD_pred =  (axial_diffusivity + radial_diffusivity + radial_diffusivity) / 3
	
	# rgb pred image
	height = restored_param.shape[0]
	width = restored_param.shape[1]
	
	theta = ang1_pred 
	phi = ang2_pred
	
	# From spherical coordinates to Cartesian coordinates
	x = np.sin(theta) * np.cos(phi)
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(theta)
	
	# Take the absolute values of the Cartesian coordinates
	x_abs = np.abs(x)
	y_abs = np.abs(y)
	z_abs = np.abs(z)
	
	# Create an RGB map with red, blue, and green
	rgb_map_pred = np.zeros((height, width,  params_test.shape[2], 3), dtype=np.uint8)
	rgb_map_pred[:, :, :,0] = x_abs * 255   # Red channel
	rgb_map_pred[:, :, :,1] = y_abs * 255   # Green channel
	rgb_map_pred[:, :, :,2] = z_abs * 255   # Blue channel
	
	rgb_map_pred[:,:,:,0] = np.where(mask_test == 1, rgb_map_pred[:,:,:,0], np.nan)
	rgb_map_pred[:,:,:,1] = np.where(mask_test == 1, rgb_map_pred[:,:,:,1], np.nan)
	rgb_map_pred[:,:,:,2] = np.where(mask_test == 1, rgb_map_pred[:,:,:,2], np.nan)
	
	rgb_map_pred2 = np.zeros((height,width, mask_test.shape[2], 3), dtype=np.uint8)
	rgb_map_pred2[:, :, :,0] = rgb_map_pred[:, :, :, 0] * FA_pred   # Red channel
	rgb_map_pred2[:, :, :,1] = rgb_map_pred[:, :, :, 1] * FA_pred   # Green channel
	rgb_map_pred2[:, :, :,2] = rgb_map_pred[:, :, :, 2] * FA_pred   # Blue channel
	
	
	### plot images
	
	num = 0
	
	plt.imshow(true_test_params_unvec_1[:,:,3]-true_test_params_unvec[:,:,num,3])
	plt.show()
	
	plt.figure(figsize=(14, 9), facecolor='black')
	plt.suptitle('Results from LS optimization', fontsize=20, color='white')
	plt.subplot(2, 3, 1)
	plt.imshow(np.rot90(true_test_params_unvec[:,:,num,4]), cmap='gray')
	plt.title('S0', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 2)
	plt.imshow(np.rot90(true_test_params_unvec[:,:,num,2]), cmap='gray', vmin = 0, vmax = 3.2)
	plt.title('AD/d_par', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 3)
	plt.imshow(np.rot90(true_test_params_unvec[:,:,num,3]), cmap='gray', vmin = 0, vmax = 3.2)
	plt.title('RD/d_per', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 4)
	plt.imshow(np.rot90(FA_true[:,:,num]), cmap='gray')
	plt.title('FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 5)
	plt.imshow(np.rot90(MD_true[:,:,num]), cmap='gray', vmin = 0, vmax = 3.2 )
	plt.title('MD', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 6)
	plt.imshow(np.rot90(rgb_map_true2[:,:,num]), cmap='jet')
	plt.title('rgb_map * FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.show()
	
	
	plt.figure(figsize=(14, 9), facecolor='black')
	plt.suptitle('Results from LS optimization', fontsize=20, color='white')
	plt.subplot(2, 3, 1)
	plt.imshow(np.rot90(true_test_params_unvec[:,:,num,4]), cmap='gray')
	plt.title('S0', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 2)
	plt.imshow(np.rot90(true_test_params_unvec[:,:,num,2]), cmap='gray')
	plt.title('AD/d_par', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 3)
	plt.imshow(np.rot90(true_test_params_unvec[:,:,num,3]), cmap='gray')
	plt.title('RD/d_per', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 4)
	plt.imshow(np.rot90(FA_true[:,:,num]), cmap='gray')
	plt.title('FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 5)
	plt.imshow(np.rot90(MD_true[:,:,num]), cmap='gray')
	plt.title('MD', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	plt.subplot(2, 3, 6)
	plt.imshow(np.rot90(rgb_map_true2[:,:,num]), cmap='jet')
	plt.title('rgb_map * FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.show()
	
	
	
	
	plt.figure(figsize=(14, 9), facecolor='black')
	plt.suptitle('Results from fitting with ML', fontsize=25, color='white')
	
	plt.subplot(2, 3, 1)
	plt.imshow(np.rot90(S0_pred[:,:,num]), cmap='gray')
	plt.title('S0', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 2)
	plt.imshow(np.rot90(AD_pred[:,:,num]), cmap='gray', vmin = 0, vmax = 3.2)
	plt.title('AD/d_par', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 3)
	plt.imshow(np.rot90(RD_pred[:,:,num]), cmap='gray', vmin = 0, vmax = 3.2)
	plt.title('RD/d_per', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 4)
	plt.imshow(np.rot90(FA_pred[:,:,num]), cmap='gray')
	plt.title('FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 5)
	plt.imshow(np.rot90(MD_pred[:,:,num]), cmap='gray', vmin = 0, vmax = 3.2 )
	plt.title('MD', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 6)
	plt.imshow(np.rot90(rgb_map_pred2[:,:,num]), cmap='jet')
	plt.title('rgb_map * FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.show()
	
	
	
	
	plt.figure(figsize=(14, 9), facecolor='black')
	plt.suptitle('Results from fitting with ML', fontsize=25, color='white')
	
	plt.subplot(2, 3, 1)
	plt.imshow(np.rot90(S0_pred[:,:,num]), cmap='gray')
	plt.title('S0', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 2)
	plt.imshow(np.rot90(AD_pred[:,:,num]), cmap='gray')
	plt.title('AD/d_par', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 3)
	plt.imshow(np.rot90(RD_pred[:,:,num]), cmap='gray')
	plt.title('RD/d_per', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 4)
	plt.imshow(np.rot90(FA_pred[:,:,num]), cmap='gray')
	plt.title('FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 5)
	plt.imshow(np.rot90(MD_pred[:,:,num]), cmap='gray')
	plt.title('MD', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.subplot(2, 3, 6)
	plt.imshow(np.rot90(rgb_map_pred2[:,:,num]), cmap='jet')
	plt.title('rgb_map * FA map', color='white',fontsize=15)
	plt.axis('off')
	color_bar = plt.colorbar()                          
	cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
	plt.setp(cbytick_obj, color='white')
	
	plt.show()
	
	
	####### COMPARE THEM ######
	rgb_map_diff = np.empty((rgb_map_pred2.shape[0], rgb_map_pred2.shape[1], rgb_map_pred2.shape[2]))
	rgb_map_diff3 = np.empty((rgb_map_true2.shape[0], rgb_map_true2.shape[1], rgb_map_true2.shape[2]))
	
	dot_product = np.empty((rgb_map_pred2.shape[0], rgb_map_pred2.shape[1], rgb_map_pred2.shape[2]))
	for dim in range(0,rgb_map_pred2.shape[2],1):
	    for pix_x in range(0,rgb_map_pred2.shape[0],1):
	        for pix_y in range(0,rgb_map_pred2.shape[1],1):
	            vector1 = np.array(rgb_map_pred2[pix_x, pix_y, dim, :])
	            vector1 = np.double(vector1)
	            vector2 = np.array(rgb_map_true2[pix_x, pix_y, dim, :])
	            vector2 = np.double(vector2)
	            dot_product[pix_x,pix_y, dim] = np.dot(vector1, vector2)
	            rgb_map_diff[pix_x,pix_y, dim] = (dot_product[pix_x,pix_y, dim] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
	
	rgb_map_diff2 = 1 - rgb_map_diff
	
	
	
	#### try with mask with csf out
	
	
	# =============================================================================
	# # Paths
	# folder_path = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_100307_4_'
	# file_path_nii_GT_signal = folder_path + 'dwi_noised_GNL.nii'
	# S, affine_estimated_params = load_nifti(file_path_nii_GT_signal)
	# print('S shape: ', S.shape)
	# 
	# file_path_mask = folder_path + 'mask.bin'
	# with open(file_path_mask, 'rb') as file:
	#     file_name = os.path.basename(file.name)
	#     mask_binary = pk.load(file)  
	# print('mask_binary: ',mask_binary.shape)
	# 
	# # =============================================================================
	# # from skimage import exposure, io
	# # # Perform intensity normalization using histogram equalization
	# # normalized_image = exposure.equalize_hist(S[:,:,70,0])
	# # 
	# # # Display original and normalized images
	# # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	# # axes[0].imshow(S[:,:,70,0], cmap='gray')
	# # axes[0].set_title('Original Image')
	# # axes[0].axis('off')
	# # axes[1].imshow(normalized_image, cmap='gray')
	# # axes[1].set_title('Normalized Image')
	# # axes[1].axis('off')
	# # plt.show()
	# # =============================================================================
	# 
	# num = 75
	# 
	# 
	# plt.imshow(np.rot90(mask_binary[:,:,num]))
	# plt.show()
	# 
	# plt.imshow(np.rot90(S[:,:,num,0]), cmap='gray')
	# plt.colorbar()
	# plt.show()
	# 
	# threshold = 7000
	# mask2 = S[:,:,:,0] > threshold
	# mask_test2 = mask_binary ^ mask2
	# 
	# plt.imshow(np.rot90(S[:,:,num,0]), cmap='gray')
	# plt.colorbar()
	# plt.imshow(np.rot90(mask_test2[:,:,num]), alpha =0.2)
	# plt.colorbar()
	# plt.axis('off')
	# plt.show()
	# 
	# plt.imshow(np.rot90(S[:,:,num,0]* (mask_test2[:,:,num])), cmap='gray')
	# plt.colorbar()
	# plt.show()
	# 
	# plt.imshow(np.rot90(S[:,:,num,0]), cmap='gray')
	# plt.colorbar()
	# plt.show()
	# 
	# 
	# num=0
	# 
	# mask_test2 = mask_test2[:,:,70:77]
	# =============================================================================
	
	mask_test2 = mask_test
	
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
	fig.suptitle('Plot difference ML - GT. csf masked', fontsize=25)
	im = axes[0,0].imshow((np.rot90(S0_pred[:,:,num] * mask_test2[:,:,num]- np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test2[:,:,num]))), cmap='gray')
	axes[0,0].set_title('S0 diff', fontsize=15)
	axes[0,0].axis('off')
	fig.colorbar(im, ax=axes[0,0])
	im = axes[0,1].imshow(np.rot90(AD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2]* mask_test2[:,:,num])), cmap='gray')
	axes[0,1].set_title('AD diff', fontsize=15)
	axes[0,1].axis('off')
	fig.colorbar(im, ax=axes[0,1])
	im = axes[0,2].imshow(np.rot90(RD_pred[:,:,num] * mask_test2[:,:,num]- np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test2[:,:,num])), cmap='gray')
	axes[0,2].set_title('RD diff', fontsize=15)
	axes[0,2].axis('off')
	fig.colorbar(im, ax=axes[0,2])
	im = axes[1,0].imshow(np.rot90(FA_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(FA_true[:,:,num]* mask_test2[:,:,num])), cmap='gray')
	axes[1,0].set_title('FA diff', fontsize=15)
	axes[1,0].axis('off')
	fig.colorbar(im, ax=axes[1,0])
	im = axes[1,1].imshow(np.rot90(MD_pred[:,:,num] * mask_test2[:,:,num]- np.nan_to_num(MD_true[:,:,num]* mask_test2[:,:,num])), cmap='gray')
	axes[1,1].set_title('MD diff', fontsize=15)
	axes[1,1].axis('off')
	fig.colorbar(im, ax=axes[1,1])
	im = axes[1,2].imshow(np.rot90(np.nan_to_num(rgb_map_diff2[:,:,num]* mask_test2[:,:,num])), cmap='gray')
	axes[1,2].set_title('rgb_map diff', fontsize=15)
	axes[1,2].axis('off')
	fig.colorbar(im, ax=axes[1,2])
	plt.show()
	
	
	restored_signal_AD = np.abs(AD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2])* mask_test2[:,:,num])
	restored_signal_RD = np.abs(RD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3])* mask_test2[:,:,num])
	restored_signal_FA = np.abs(FA_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(FA_true[:,:,num]* mask_test2[:,:,num]))
	restored_signal_MD = np.abs(MD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(MD_true[:,:,num]* mask_test2[:,:,num]))
	angles = np.nan_to_num(rgb_map_diff2[:,:,num]* mask_test2[:,:,num])
	restored_S0 = np.abs(S0_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test2[:,:,num]))
	
	vmin = min(restored_signal_AD.min(), restored_signal_RD.min(), restored_signal_FA.min(), restored_signal_MD.min())
	vmax = max(restored_signal_AD.max(), restored_signal_RD.max(), restored_signal_FA.max(), restored_signal_MD.max())
	
	fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(21, 4))
	img1 = axes[0].imshow(np.rot90(restored_signal_AD), cmap='jet', vmin=vmin, vmax=vmax)
	img2 = axes[1].imshow(np.rot90(restored_signal_RD), cmap='jet', vmin=vmin, vmax=vmax)
	img3 = axes[2].imshow(np.rot90(restored_signal_FA), cmap='jet', vmin=vmin, vmax=vmax)
	img4 = axes[3].imshow(np.rot90(restored_signal_MD), cmap='jet', vmin=vmin, vmax=vmax)
	axes[0].axis('off')
	axes[1].axis('off')
	axes[2].axis('off')
	axes[3].axis('off')
	cbar = fig.colorbar(img4, ax=axes[3], orientation='vertical')
	axes[0].set_title('AD ')
	axes[1].set_title('RD')
	axes[2].set_title('FA')
	axes[3].set_title('MD')
	img5 = axes[4].imshow(np.rot90(angles), cmap='jet')
	cbar = fig.colorbar(img5, ax=axes[4], orientation='vertical')
	axes[4].axis('off')
	axes[4].set_title('angles')
	img6 = axes[5].imshow(np.rot90(restored_S0), cmap='jet')
	cbar = fig.colorbar(img6, ax=axes[5], orientation='vertical')
	axes[5].axis('off')
	axes[5].set_title('S0')
	plt.suptitle('Parameter differences csf masked', fontsize=20)
	plt.tight_layout()
	plt.show()
	
	
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9), facecolor='black')
	fig.suptitle('Plot difference abs(ML-LS) without csf', fontsize=25, color='white')
	
	im = axes[0,0].imshow(np.rot90(np.abs(S0_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test2[:,:,num]))), cmap='jet')
	axes[0,0].set_title('S0 diff', fontsize=14, color='white')
	axes[0,0].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 0])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[0,1].imshow(np.rot90(np.abs(AD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2])* mask_test2[:,:,num])), cmap='jet')
	axes[0,1].set_title('AD diff', fontsize=15,  color='white')
	axes[0,1].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 1])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[0,2].imshow(np.rot90(np.abs(RD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test2[:,:,num]))), cmap='jet')
	axes[0,2].set_title('RD diff', fontsize=15,  color='white')
	axes[0,2].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 2])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[1,0].imshow(np.rot90(np.abs(FA_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(FA_true[:,:,num]* mask_test2[:,:,num]))), cmap='jet')
	axes[1,0].set_title('FA diff', fontsize=15,  color='white')
	axes[1,0].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 0])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[1,1].imshow(np.rot90(np.abs(MD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(MD_true[:,:,num]* mask_test2[:,:,num]))), cmap='jet')
	axes[1,1].set_title('MD diff', fontsize=15,  color='white')
	axes[1,1].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 1])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[1,2].imshow(np.rot90(np.nan_to_num(rgb_map_diff2[:,:,num]* mask_test2[:,:,num])), cmap='jet')
	axes[1,2].set_title('rgb_map diff. 1-|a.b|', fontsize=15,  color='white')
	axes[1,2].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 2])
	cbar.ax.tick_params(labelcolor='white') 
	
	plt.show()
	
	per = 99
	
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9), facecolor='black')
	fig.suptitle(f'Plot difference abs(ML-LS) without csf percentile {per} ', fontsize=25, color='white')
	
	image1 =np.rot90(np.abs(S0_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test2[:,:,num])))
	image2 = np.percentile(image1, per)
	clipped_img = np.clip(image1, None, image2)
	
	im = axes[0,0].imshow(clipped_img, cmap='jet')
	axes[0,0].set_title('S0 diff', fontsize=14, color='white')
	axes[0,0].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 0])
	cbar.ax.tick_params(labelcolor='white') 
	
	image1 =np.rot90(np.abs(AD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2]* mask_test2[:,:,num])))
	image2 = np.percentile(image1, per)
	clipped_img = np.clip(image1, None, image2)
	
	im = axes[0,1].imshow(clipped_img, cmap='jet')
	axes[0,1].set_title('AD diff', fontsize=15,  color='white')
	axes[0,1].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 1])
	cbar.ax.tick_params(labelcolor='white') 
	
	image1 =np.rot90(np.abs(RD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test2[:,:,num])))
	image2 = np.percentile(image1, per)
	clipped_img = np.clip(image1, None, image2)
	
	im = axes[0,2].imshow(clipped_img, cmap='jet')
	axes[0,2].set_title('RD diff', fontsize=15,  color='white')
	axes[0,2].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 2])
	cbar.ax.tick_params(labelcolor='white') 
	
	image1 = np.rot90(np.abs(FA_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(FA_true[:,:,num]* mask_test2[:,:,num])))
	image2 = np.percentile(image1, per)
	clipped_img = np.clip(image1, None, image2)
	
	im = axes[1,0].imshow(clipped_img, cmap='jet')
	axes[1,0].set_title('FA diff', fontsize=15,  color='white')
	axes[1,0].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 0])
	cbar.ax.tick_params(labelcolor='white') 
	
	image1 = np.rot90(np.abs(MD_pred[:,:,num]* mask_test2[:,:,num] - np.nan_to_num(MD_true[:,:,num]* mask_test2[:,:,num])))
	image2 = np.percentile(image1, per)
	clipped_img = np.clip(image1, None, image2)
	
	im = axes[1,1].imshow(clipped_img, cmap='jet')
	axes[1,1].set_title('MD diff', fontsize=15,  color='white')
	axes[1,1].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 1])
	cbar.ax.tick_params(labelcolor='white')
	
	image1 = np.rot90(np.nan_to_num(rgb_map_diff2[:,:,num]* mask_test2[:,:,num]))
	image2 = np.percentile(image1, per)
	clipped_img = np.clip(image1, None, image2)
	
	im = axes[1,2].imshow(clipped_img, cmap='jet')
	axes[1,2].set_title('rgb_map diff. 1-|a.b|', fontsize=15,  color='white')
	axes[1,2].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 2])
	cbar.ax.tick_params(labelcolor='white') 
	
	plt.show()
	
	
	
	
	## plot images
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
	fig.suptitle('Plot difference ML - GT', fontsize=25)
	
	im = axes[0,0].imshow((np.rot90(S0_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test[:,:,num]))), cmap='gray')
	axes[0,0].set_title('S0 diff', fontsize=15)
	axes[0,0].axis('off')
	fig.colorbar(im, ax=axes[0,0])
	
	im = axes[0,1].imshow(np.rot90(AD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2]* mask_test[:,:,num])), cmap='gray')
	axes[0,1].set_title('AD diff', fontsize=15)
	axes[0,1].axis('off')
	fig.colorbar(im, ax=axes[0,1])
	
	im = axes[0,2].imshow(np.rot90(RD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test[:,:,num])), cmap='gray')
	axes[0,2].set_title('RD diff', fontsize=15)
	axes[0,2].axis('off')
	fig.colorbar(im, ax=axes[0,2])
	
	im = axes[1,0].imshow(np.rot90(FA_pred[:,:,num] - np.nan_to_num(FA_true[:,:,num])), cmap='gray')
	axes[1,0].set_title('FA diff', fontsize=15)
	axes[1,0].axis('off')
	fig.colorbar(im, ax=axes[1,0])
	
	im = axes[1,1].imshow(np.rot90(MD_pred[:,:,num] - np.nan_to_num(MD_true[:,:,num])), cmap='gray')
	axes[1,1].set_title('MD diff', fontsize=15)
	axes[1,1].axis('off')
	fig.colorbar(im, ax=axes[1,1])
	
	im = axes[1,2].imshow(np.rot90(np.nan_to_num(rgb_map_diff2[:,:,num])), cmap='gray')
	axes[1,2].set_title('rgb_map diff', fontsize=15)
	axes[1,2].axis('off')
	fig.colorbar(im, ax=axes[1,2])
	
	plt.show()
	
	
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9))
	fig.suptitle('Plot difference abs(ML-LS)', fontsize=25)
	
	im = axes[0,0].imshow(np.rot90(np.abs(S0_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test[:,:,num]))), cmap='gray')
	axes[0,0].set_title('S0 diff', fontsize=14)
	axes[0,0].axis('off')
	fig.colorbar(im, ax=axes[0,0])
	
	im = axes[0,1].imshow(np.rot90(np.abs(AD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2])* mask_test[:,:,num])), cmap='gray')
	axes[0,1].set_title('AD diff', fontsize=15)
	axes[0,1].axis('off')
	fig.colorbar(im, ax=axes[0,1])
	
	im = axes[0,2].imshow(np.rot90(np.abs(RD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test[:,:,num]))), cmap='gray')
	axes[0,2].set_title('RD diff', fontsize=15)
	axes[0,2].axis('off')
	fig.colorbar(im, ax=axes[0,2])
	
	im = axes[1,0].imshow(np.rot90(np.abs(FA_pred[:,:,num] - np.nan_to_num(FA_true[:,:,num]))), cmap='gray')
	axes[1,0].set_title('FA diff', fontsize=15)
	axes[1,0].axis('off')
	fig.colorbar(im, ax=axes[1,0])
	
	im = axes[1,1].imshow(np.rot90(np.abs(MD_pred[:,:,num] - np.nan_to_num(MD_true[:,:,num]))), cmap='gray')
	axes[1,1].set_title('MD diff', fontsize=15)
	axes[1,1].axis('off')
	fig.colorbar(im, ax=axes[1,1])
	
	im = axes[1,2].imshow(np.rot90(np.nan_to_num(rgb_map_diff2[:,:,num])), cmap='gray')
	axes[1,2].set_title('rgb_map diff. 1-|a.b|', fontsize=15)
	axes[1,2].axis('off')
	fig.colorbar(im, ax=axes[1,2])
	
	plt.show()
	
	
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 9), facecolor='black')
	fig.suptitle('Plot difference abs(ML-LS)', fontsize=25, color='white')
	
	im = axes[0,0].imshow(np.rot90(np.abs(S0_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test[:,:,num]))), cmap='jet')
	axes[0,0].set_title('S0 diff', fontsize=14, color='white')
	axes[0,0].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 0])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[0,1].imshow(np.rot90(np.abs(AD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2])* mask_test[:,:,num])), cmap='jet')
	axes[0,1].set_title('AD diff', fontsize=15,  color='white')
	axes[0,1].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 1])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[0,2].imshow(np.rot90(np.abs(RD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test[:,:,num]))), cmap='jet')
	axes[0,2].set_title('RD diff', fontsize=15,  color='white')
	axes[0,2].axis('off')
	cbar = fig.colorbar(im, ax=axes[0, 2])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[1,0].imshow(np.rot90(np.abs(FA_pred[:,:,num] - np.nan_to_num(FA_true[:,:,num]))), cmap='jet')
	axes[1,0].set_title('FA diff', fontsize=15,  color='white')
	axes[1,0].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 0])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[1,1].imshow(np.rot90(np.abs(MD_pred[:,:,num] - np.nan_to_num(MD_true[:,:,num]))), cmap='jet')
	axes[1,1].set_title('MD diff', fontsize=15,  color='white')
	axes[1,1].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 1])
	cbar.ax.tick_params(labelcolor='white') 
	
	im = axes[1,2].imshow(np.rot90(np.nan_to_num(rgb_map_diff2[:,:,num])), cmap='jet')
	axes[1,2].set_title('rgb_map diff. 1-|a.b|', fontsize=15,  color='white')
	axes[1,2].axis('off')
	cbar = fig.colorbar(im, ax=axes[1, 2])
	cbar.ax.tick_params(labelcolor='white') 
	
	plt.show()
	
	
	restored_signal_AD = np.abs(AD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2])* mask_test[:,:,num])
	restored_signal_RD = np.abs(RD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3])* mask_test[:,:,num])
	restored_signal_FA = np.abs(FA_pred[:,:,num] - np.nan_to_num(FA_true[:,:,num]))
	restored_signal_MD = np.abs(MD_pred[:,:,num] - np.nan_to_num(MD_true[:,:,num]))
	angles = np.nan_to_num(rgb_map_diff2[:,:,num])
	restored_S0 = np.abs(S0_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test[:,:,num]))
	
	vmin = min(restored_signal_AD.min(), restored_signal_RD.min(), restored_signal_FA.min(), restored_signal_MD.min())
	vmax = max(restored_signal_AD.max(), restored_signal_RD.max(), restored_signal_FA.max(), restored_signal_MD.max())
	
	fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(21, 4))
	img1 = axes[0].imshow(np.rot90(restored_signal_AD), cmap='jet', vmin=vmin, vmax=vmax)
	img2 = axes[1].imshow(np.rot90(restored_signal_RD), cmap='jet', vmin=vmin, vmax=vmax)
	img3 = axes[2].imshow(np.rot90(restored_signal_FA), cmap='jet', vmin=vmin, vmax=vmax)
	img4 = axes[3].imshow(np.rot90(restored_signal_MD), cmap='jet', vmin=vmin, vmax=vmax)
	axes[0].axis('off')
	axes[1].axis('off')
	axes[2].axis('off')
	axes[3].axis('off')
	cbar = fig.colorbar(img4, ax=axes[3], orientation='vertical')
	axes[0].set_title('AD ')
	axes[1].set_title('RD')
	axes[2].set_title('FA')
	axes[3].set_title('MD')
	
	img5 = axes[4].imshow(np.rot90(angles), cmap='jet')
	cbar = fig.colorbar(img5, ax=axes[4], orientation='vertical')
	axes[4].axis('off')
	axes[4].set_title('angles')
	
	img6 = axes[5].imshow(np.rot90(restored_S0), cmap='jet')
	cbar = fig.colorbar(img6, ax=axes[5], orientation='vertical')
	axes[5].axis('off')
	axes[5].set_title('S0')
	
	plt.suptitle('Parameter differences', fontsize=20)
	plt.tight_layout()
	plt.show()
	
	
	S0_diff = abs(S0_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,4]* mask_test[:,:,num]))
	AD_diff = abs(AD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,2]* mask_test[:,:,num]))
	RD_diff = abs(RD_pred[:,:,num] - np.nan_to_num(true_test_params_unvec[:,:,num,3]* mask_test[:,:,num]))
	FA_diff = abs(FA_pred[:,:,num] - np.nan_to_num(FA_true[:,:,num]))
	MD_diff = abs(MD_pred[:,:,num] - np.nan_to_num(MD_true[:,:,num]))
	Angles_diff = abs(np.nan_to_num(rgb_map_diff2[:,:,num]))
	
	per = 99
	
	# plot histograms 
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
	fig.suptitle(f'Plot hist of differences. {per}th per', fontsize=15)
	
	S0_hist = abs(S0_diff[mask_test[:,:,num] == 1])
	threshold_value = np.percentile(S0_hist, per)
	S0_hist_thresholded = np.clip(S0_hist, 0, threshold_value)
	axes[0, 0].hist(S0_hist_thresholded, color='b', bins=30)
	axes[0, 0].set_title('S0')
	axes[0, 0].set_xlabel('S0')
	axes[0, 0].set_ylabel('Frequency')
	
	d_par_hist = abs(AD_diff[mask_test[:,:,num] == 1])
	threshold_value = np.percentile(d_par_hist, per)
	AD_hist_thresholded = np.clip(d_par_hist, 0, threshold_value)
	axes[0, 1].hist(AD_hist_thresholded, color='b', bins=30)
	axes[0, 1].set_title('d_|| or AD [μm^2/ms]')
	axes[0, 1].set_xlabel('d_||')
	axes[0, 1].set_ylabel('Frequency')
	
	d_per_hist = abs(RD_diff[mask_test[:,:,num] == 1])
	threshold_value = np.percentile(d_per_hist, per)
	RD_hist_thresholded = np.clip(d_per_hist, 0, threshold_value)
	axes[0, 2].hist(RD_hist_thresholded, color='b', bins=30)
	axes[0, 2].set_title('d_⊥ or RD [μm^2/ms]')
	axes[0, 2].set_xlabel('d_⊥')
	axes[0, 2].set_ylabel('Frequency')
	
	FA_hist = abs(FA_diff[mask_test[:,:,num] == 1])
	threshold_value = np.percentile(FA_hist, per)
	FA_hist_thresholded = np.clip(FA_hist, 0, threshold_value)
	axes[1, 0].hist(FA_hist_thresholded, color='b', bins=30)
	axes[1, 0].set_title('FA map')
	axes[1, 0].set_xlabel('FA')
	axes[1, 0].set_ylabel('Frequency')
	
	MD_hist = abs(MD_diff[mask_test[:,:,num] == 1])
	threshold_value = np.percentile(MD_hist, per)
	MD_hist_thresholded = np.clip(MD_hist, 0, threshold_value)
	axes[1, 1].hist(MD_hist_thresholded, color='b', bins=30)
	axes[1, 1].set_title('MD [μm^2/ms]')
	axes[1, 1].set_xlabel('MD')
	axes[1, 1].set_ylabel('Frequency')
	
	ang_hist = abs(Angles_diff[mask_test[:,:,num] == 1])
	threshold_value = np.percentile(ang_hist, per)
	ang_hist_thresholded = np.clip(ang_hist, 0, threshold_value)
	axes[1, 2].hist(ang_hist_thresholded, color='b', bins=30)
	axes[1, 2].set_title('Angles [º]')
	axes[1, 2].set_xlabel('Angles')
	axes[1, 2].set_ylabel('Frequency')
	
	plt.tight_layout()
	plt.show()

