# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:05:30 2023

@author: pcastror

Visualization of synthesised data from synsignals.py

"""




import matplotlib.pyplot as plt
import pickle as pk
import os
import numpy as np

print(np.sin(2*np.pi))
print(np.sin(np.pi))
print(np.sin(0))
print(np.sin(np.deg2rad(359)))


#outstr= r'C:\Users\pcastror\Desktop\internship\ML_network_withTE'
outstr = r'C:\Users\pcastror\Desktop\internship\ML_network\ML_network_Zeppelin_MRImodelZeppelin_ntrain10000_nval3500_ntest1500_SNR[70.]\HCP'


#outstr= r'C:\Users\pcastror\Desktop\internship\ML_network'
num = 50  ## signal number to plot. note that depending on the set i will have different bounds. the one implemented: stest (10,288)

with open(r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\bvals.txt', 'r') as f:
    protocol_content = [line.rstrip('\n') for line in f]

############### plot synthetic PARAMETERS from MRI signals  ###################
with open('{}_paramtrain.bin'.format(outstr), 'rb') as file:
    file_name = os.path.basename(file.name)
    data = pk.load(file)

# Verify the size of the bin file
print('Size of the *.bin file: ', data.shape)
print('data.shape[1]: ', data.shape[1]) 
    
   
plt.plot(data[:,0])
plt.title('polar')
plt.show()

plt.plot(data[:,1])
plt.title('azimutal')
plt.show()

print(data[:,1].shape)
print(data[1,1])


Parameters = ['ang(1)', 'ang(2)', 'd_z_par', 'd_z_per', 'S0']
#Parameters = ['ang(1)', 'ang(1)','ang(1)','ang(1)','ang(1)', 'ang(2)', 'd_z_par', 'd_z_per', 'S0']

# Create histogram for parameters
fig, axs = plt.subplots(1, data.shape[1], figsize=(15, 3))
fig.subplots_adjust(hspace=0.5)
fig.suptitle(file_name, fontsize=16)
# =============================================================================
# legend_text = ['Parameters: [ang(1), ang(2), d_z_par, d_z_per, S0]']
# custom_legend = [plt.Line2D([0], [0], color='blue', lw=2, label=text) for text in legend_text]
# fig.legend(custom_legend, legend_text, loc='upper left', fontsize = 10)
# =============================================================================
for i in range(0,data.shape[1]):
    # Boxplot
    axs[i].hist(data[:, i], bins=50, density=True, color='b')
    axs[i].set_xlabel('Value', fontsize = 10)
    axs[i].set_ylabel('Frequency', fontsize = 10)
    axs[i].set_title(f'{Parameters[i]}', fontsize = 10)


# Show the plot
plt.tight_layout()
plt.show()


############### plot synthetic MRI SIGNALS WITHOUT NOISE ###################
# Open the binary file in binary mode for reading
with open('{}_sigtrain_NOnoise.bin'.format(outstr), 'rb') as file:
    print(file)
    file_name = os.path.basename(file.name)
    data_no_noise = pk.load(file)

# Verify the size of the bin file
#print('Size of the *.bin file: ', data_no_noise.shape)
#print('Size of the *.bin file: ', data_no_noise)

with open(r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\bvals.txt', 'r') as f:
    protocol_content = [line.rstrip('\n') for line in f]

i = len(data_no_noise)
for arrays in data_no_noise:
    plt.plot(arrays)

# Show the plot with all the signals generated
plt.tight_layout()
legend_text = ['Number of signals: ' + str(i), 'b-vals: ' + str(protocol_content[0])]
custom_legend = [plt.Line2D([0], [0], color='none', lw=1, label=text) for text in legend_text]
plt.legend(custom_legend, legend_text, loc='upper left',  fontsize='x-small')
plt.title(f'MRI SIGNALS WITHOUT NOISE: {file_name}')
plt.show()


# plot only the first signal for example
plt.plot(data_no_noise[num])
plt.tight_layout()
plt.title(f'MRI SIGNALS WITHOUT NOISE [{num}]: {file_name}')
plt.show()



############### plot synthetic MRI SIGNALS WITH NOISE ###################
# Open the binary file in binary mode for reading
with open('{}_sigtrain.bin'.format(outstr), 'rb') as file:
    print(file)
    file_name = os.path.basename(file.name)
    data_noise = pk.load(file)

# Verify the size of the bin file
#print('Size of the *.bin file: ', data_noise.shape)
#print('Size of the *.bin file: ', data_noise)

nvox_train = data_noise.shape[0]    # number of voxels
nmeas_train = data_noise.shape[1]   # number of measurements

print(nvox_train)
print(nmeas_train)


i = len(data_noise)
for arrays in data_noise:
    plt.plot(arrays)

# Show the plot with all the signals generated
plt.tight_layout()
legend_text = ['Number of signals: ' + str(i), 'b-vals: ' + str(protocol_content[0])]
custom_legend = [plt.Line2D([0], [0], color='none', lw=1, label=text) for text in legend_text]
plt.legend(custom_legend, legend_text, loc='upper left',  fontsize='x-small')
plt.title(f'MRI SIGNALS WITH NOISE: {file_name}')
plt.show()


# plot only the first signal for example
plt.plot(data_noise[num])
plt.tight_layout()
plt.title(f'MRI SIGNALS WITH NOISE [{num}]: {file_name}')
plt.show()

print(data_noise[num].shape)

########## plot with and wothout noise to appreciate the difference


plt.plot(data_no_noise[num], label='no noise')
plt.plot(data_noise[num], label='noised')
plt.legend()
plt.title(f'SIGNALS WITH AND WITHOUT NOISE strain[{num}]')

plt.tight_layout()
plt.show()

plt.plot(np.abs(data_no_noise[num]-data_noise[num]))
plt.title(f'| strain[{num}] - strain_noisy[{num}] |')

plt.tight_layout()
plt.show()



############### plot voxel-wise noise levels (SNR values)  ###################
# =============================================================================
# # Open the binary file in binary mode for reading
# with open('{}_sigmatrain.bin'.format(outstr), 'rb') as file:
#     file_name = os.path.basename(file.name)
#     # Load the object from the file
#     data = pk.load(file)
# 
# # Verify the size of the bin file
# print('Size of the *.bin file: ', data.shape)
# 
# 
# # Display the values
# # for value in data:
# #    print(value)
#     
# print('data.shape[1]: ', data.shape[1])
# 
# # Histogram
# plt.hist(data[:], bins=15)
# plt.xlabel(f'Parameter {i+1}')
# plt.ylabel('Frequency')
# plt.title(f'Voxel-wise noise levels: {file_name}')
#     
# # Show the plot
# plt.tight_layout()
# plt.show()
# =============================================================================


import numpy as np

nneurons = [5, 7, 9]
print('** Hidden neurons: {}'.format(nneurons))

nlayers = np.array(nneurons[1])  # Number of layers

print('nlayers: ', nlayers)
nneurons = np.array(nneurons)          # Number of hidden neurons in each layer

print('nneurons: ', nneurons)


    
    



