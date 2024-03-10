# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:19:49 2023

@author: pcastror

Visualize training SIMULATION results from train_signals 

"""


import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from scipy import stats


path_pred = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\ML_strat2\results_Exp1_nhidden108-56-5_pdrop0.0_noepoch300_lr0.0001_mbatch100_seed12345'
path_true = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\Simulations\exp1_MRImodelZeppelin_ntrain8000_nval2000_ntest1000_SNR70\syn_GT'


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


print(true_val_signal.shape)
print(true_val_param.shape)

print(true_test_signal.shape)
print(true_test_param.shape)


with open( '{}\epoch300_net.bin'.format(path_pred), 'rb') as file:
    model = pk.load(file)

# Verify the size of the bin file
print('Model at epoch 300: ', model)
    
with open('{}\losstrain.bin'.format(path_pred), 'rb') as file:
    file_losstrain = os.path.basename(file.name)
    loss_train = pk.load(file)

with open('{}\lossval.bin'.format(path_pred), 'rb') as file:
    file_lossval = os.path.basename(file.name)
    loss_val = pk.load(file)

  
    
# Verify the size of the bin file
print('Size of the training loss: ', loss_train.shape)
print('Size of the validation loss: ', loss_val.shape)
print('Number of epoch: ', loss_train.shape[0])
print('Number of mini-batches in train loss: ', loss_train.shape[1])
print('Number of mini-batches in validation loss: ', loss_val.shape[1])

epochs = loss_train.shape[0]

# average loss from mini_batches to plot
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
axes[0].set_title(file_losstrain)
axes[0].grid(True)

axes[1].plot(epochs, new_loss_val)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title(file_lossval)
axes[1].grid(True)

plt.show()


fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5)

axes[0].plot(epochs, new_loss_train)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title(file_losstrain, fontsize=20)
axes[0].grid(True)

axes[1].plot(epochs, new_loss_val)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title(file_lossval , fontsize=20)
axes[1].grid(True)

plt.show()

print(new_loss_train.shape)
print(epochs)

plt.plot(epochs, new_loss_train, label=file_losstrain)
plt.plot(epochs, new_loss_val, label=file_lossval)

# Add labels and title to the plot
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Losses train vs validation')

# Add a legend
plt.legend()

# Display the plot
plt.show()

print('                               ')



################## signal losses ###############

with open('{}\lossvalmin_sigval.bin'.format(path_pred), 'rb') as file:
    file_loss_sigval = os.path.basename(file.name)
    loss_sigval = pk.load(file)

with open('{}\lossvalmin_sigtest.bin'.format(path_pred), 'rb') as file:
    file_loss_sigtest = os.path.basename(file.name)
    loss_sigtest = pk.load(file)

# Verify the size of the bin file
print('Size of the loss_sigval loss: ', loss_sigval.shape)
print('Size of the loss_sigtest loss: ', loss_sigtest.shape)
print('Number of samples loss_sigval: ', loss_sigval.shape[0])
print('Number of samples loss_sigtest: ', loss_sigtest.shape[0])
print('Number of mini-batches in loss_sigval loss: ', loss_sigval.shape[1])
print('Number of mini-batches in loss_sigtest loss: ', loss_sigtest.shape[1])

samples_sigval = loss_sigval.shape[0]
samples_sigtest = loss_sigtest.shape[0]

num = 200
#print(loss_sigval[num,:])

# plot signals
plt.plot(loss_sigval[num,:])
plt.tight_layout()
plt.title(f'MRI SIGNALS sigval [{num}]')
plt.show()

# plot signals
plt.plot(loss_sigtest[num,:])
plt.tight_layout()
plt.title(f'MRI SIGNALS sigtest [{num}]')
plt.show()


print(true_test_signal.shape)
print(loss_sigtest.shape)

print(true_val_signal.shape)
print(loss_sigval.shape)

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

# Add labels and title to the plot
plt.xlabel('value')
plt.ylabel('signal')
plt.title(f'pred_sig_test [{num}] vs true_sig_test [{num}]' )

# Add a legend
plt.legend()

print('')

################## signal tissue losses ###############

with open('{}\lossvalmin_tissueval.bin'.format(path_pred), 'rb') as file:
    # Load the object from the file
    file_loss_tissueval = os.path.basename(file.name)
    loss_tissueval = pk.load(file)
    
loss_tissueval[:,3] = loss_tissueval[:,2] * loss_tissueval[:,3]


with open('{}\lossvalmin_tissuetest.bin'.format(path_pred), 'rb') as file:
    # Load the object from the file
    file_loss_tissuetest = os.path.basename(file.name)
    loss_tissuetest = pk.load(file)

loss_tissuetest[:,3] = loss_tissuetest[:,2] * loss_tissuetest[:,3]

    
param_name = ['Ang1','Ang2','AD','RD', 'S0']

print('PARAMETERS: ', param_name)

# Verify the size of the bin file
print('Size of the loss_tissueval loss: ', loss_tissueval.shape)
print('Size of the loss_tissuetest loss: ', loss_tissuetest.shape)
print('Number of samples loss_tissueval: ', loss_tissueval.shape[0])
print('Number of samples loss_tissuetest: ', loss_tissuetest.shape[0])
print('Number of mini-batches in loss_tissueval loss: ', loss_tissueval.shape[1])
print('Number of mini-batches in loss_tissuetest loss: ', loss_tissuetest.shape[1])

parameters_tissueval = loss_tissueval.shape[1]
parameters_tissuetest = loss_tissuetest.shape[1]

print(true_val_param.shape)
print(loss_tissueval.shape)

fig, axes = plt.subplots(1, parameters_tissueval, figsize=(15, 4))
# Loop through each parameter and create a scatter plot
for param in range(parameters_tissueval):
    axes[param].hist(loss_tissueval[:, param])
    axes[param].set_xlabel('Value')
    axes[param].set_ylabel('Frequency')
    axes[param].set_title(f'{param_name [param]}')
    
fig.suptitle('Histogram of predicted parameters VALIDATION', fontsize=16)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


fig, axes = plt.subplots(1, parameters_tissuetest, figsize=(15, 4))
# Loop through each parameter and create a scatter plot
for param in range(parameters_tissuetest):
    axes[param].hist(loss_tissuetest[:, param])
    axes[param].set_xlabel('Value')
    axes[param].set_ylabel('Frequency')
    axes[param].set_title(f'{param_name [param]}')
    
fig.suptitle('Histogram of predicted parameters TEST', fontsize=16)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


## COMPARISON WITH THE GROUND TRUTH

parameters_tissueval = loss_tissueval.shape[1]
parameters_tissuetest = loss_tissuetest.shape[1]

# Scatter plots of simulation results obtained on the test set. predicted signal (y-axis) against input measurements (x-axis)
fig, axes = plt.subplots(1, parameters_tissueval-2, figsize=(10, 4))

# Loop through each parameter and create a scatter plot
for param in range(2,parameters_tissueval):
    print(param)
    axes[param-2].scatter(true_val_param[:, param], loss_tissueval[:, param], s=5)
    # Fit a linear regression line to the data to plot trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_val_param[:, param],  loss_tissueval[:, param])
    trend_line = slope * np.array(true_val_param[:, param]) + intercept
    axes[param-2].plot(true_val_param[:, param], trend_line, color='red', label='Trend Line', linestyle='--')   
    axes[param-2].set_xlabel(f'True {param_name[param]}')
    axes[param-2].set_ylabel(f'Predicted {param_name[param]}')
    axes[param-2].set_title(f'{param_name [param]}')
    #axes[param].legend()
    
fig.suptitle('Scatter Plots of True vs ML prediction VALIDATION', fontsize=16)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


# Scatter plots of simulation results obtained on the test set. predicted signal (y-axis) against input measurements (x-axis)
fig, axes = plt.subplots(1, parameters_tissuetest-2, figsize=(10, 4))
# Loop through each parameter and create a scatter plot
for param in range(2,parameters_tissuetest):
    axes[param-2].scatter(true_test_param[:, param], loss_tissuetest[:, param], s=5)
    # Fit a linear regression line to the data to plot trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_test_param[:, param],  loss_tissuetest[:, param])
    trend_line = slope * np.array(true_test_param[:, param]) + intercept
    axes[param-2].plot(true_test_param[:, param], trend_line, color='red', label='Trend Line', linestyle='--')    
    axes[param-2].set_xlabel(f'True {param_name[param]}')
    axes[param-2].set_ylabel(f'Predicted {param_name[param]}')
    axes[param-2].set_title(f'{param_name [param]}')
    #axes[param].legend()
    
fig.suptitle('Scatter Plots of True vs ML prediction TEST', fontsize=16)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


################## lossvalmin_net. ADDITIONAL INFORMATION ###############

with open('{}\lossvalmin_net.bin'.format(path_pred), 'rb') as file:
    # Load the object from the file
    file_lossvalmin_net = os.path.basename(file.name)
    lossvalmin_net = pk.load(file)
    
    print('Model at best epoch: ', lossvalmin_net, '\n')



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
    


with open('{}\lossvalmin_neuronval.bin'.format(path_pred), 'rb') as file:
    # Load the object from the file
    file_path_lossvalmin_neuronval = os.path.basename(file.name)
    lossvalmin_neuronval = pk.load(file)

with open('{}\lossvalmin_neurontest.bin'.format(path_pred), 'rb') as file:
    # Load the object from the file
    file_path_lossvalmin_neurontest = os.path.basename(file.name)
    lossvalmin_neurontest = pk.load(file)

print('lossvalmin_neuronval: The output neuron activations for validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch')
print('lossvalmin_neurontest: The output neuron activations for test signals (shape: voxels x number_of_tissue_parameters) at the best epoch ')



fig, axes = plt.subplots(1, lossvalmin_neuronval.shape[1], figsize=(15, 4))
# Loop through each parameter and create a scatter plot
for param in range(lossvalmin_neuronval.shape[1]):
    axes[param].hist(lossvalmin_neuronval[:, param])
    axes[param].set_xlabel('Value')
    axes[param].set_ylabel('Frequency')
    axes[param].set_title(f'{param_name [param]}')
    
fig.suptitle('Histogram of predicted lossvalmin_neuronval', fontsize=16)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()



fig, axes = plt.subplots(1, lossvalmin_neurontest.shape[1], figsize=(15, 4))
# Loop through each parameter and create a scatter plot
for param in range(lossvalmin_neurontest.shape[1]):
    axes[param].hist(lossvalmin_neurontest[:, param])
    axes[param].set_xlabel('Value')
    axes[param].set_ylabel('Frequency')
    axes[param].set_title(f'{param_name [param]}')
    
fig.suptitle('Histogram of predicted lossvalmin_neurontest', fontsize=16)

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


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



print('')
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



   
