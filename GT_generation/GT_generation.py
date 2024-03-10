# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:04:31 2023

@author: pcastror

This code fits images from the HCP to the Zeppelin model. Multiple initializations. 
bvals up until 108.

This code saves:
    - GT parameters as .nii file
    - residuals for each iteration performed as .nii file
    - INFO .txt file with some details of the fitting procedure to get the GT
    
    - Simulatons: GT signal with GNL included. No noise
    
    * TRAINING SETS
    - Simulation + noise + GNL
    
OPTIONAL: optional visualization of results while runnign script. 
    - for visualization: vis=1
    - for no vosualization. vis=0
    
"""

### Load libraries
import numpy as np
from scipy.optimize import least_squares
from dipy.io.image import load_nifti
import time
import nibabel as nib
import pickle as pk
import matplotlib.pyplot as plt
import torch
from utilities import vec, unvec, Zeppelin, bval_bvec_from_b_Matrix
from dipy.reconst.dti import fractional_anisotropy
from dipy.segment.mask import median_otsu


print('') 
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('Device used. using GPU (cuda) will significantly speed up the training:', device) # Device used

image_idx = '0408'

vis = 1

print('')  
print('*************************************************************************************')
print('                    PART 1: GENERATE GROUND TRUTH PARAMETERS                         ')
print('*************************************************************************************')
print('')

# Read the image
file_path_nii = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\10' + image_idx + '_DWI.nii'
file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\10' + image_idx + '_DWI.txt'
dwi, affine = load_nifti(file_path_nii)
b_matrix = np.loadtxt(file_path_bvec)

dwi_ = dwi

# extract bvals and bvecs from b matrix
bval, g = bval_bvec_from_b_Matrix (b_matrix) 
bval=bval/1000

# choose only first 108 bvalues because im using a tensor model
bval = bval[:108]
dwi = dwi[:,:,:,:108]
g = g[:108,:]

maskdata, msk = median_otsu(dwi[:,:,:,0], median_radius=6, numpass=1)

num = 80
plt.imshow(dwi[:,:,num,0], cmap='gray')
plt.imshow(msk[:,:,num], cmap='viridis', alpha=0.3)
plt.title('Masked dwi')
plt.show()

dwi = vec(dwi,msk)
dwi = dwi[0]
dwi = dwi.T

niter = 5

print('')  
print('********************************* Load data to fit **********************************')
print('')

# choose lower and upper bounds
lb = [0, 0, 0, 0, 0]
ub = [np.pi, 2*np.pi, 3.2, 1, np.inf]

print('Image shape: ', dwi.shape)
print('B-matrix shape: ', b_matrix.shape)
print('g (bvecs): ',g.shape)
print('bvals: ', bval.shape)

print('')
print
print('******************************* Fit the signal from HCP *****************************')
print('')

print('FITTING PROCEDURE INFO')
print('Lower bounds: ', lb)
print('Upper bounds: ', ub)

# extract the shape of the images. 
dwi_size = dwi.shape[0]
params_size = np.array(lb).shape[0]

# initiate the variable to store the estimated paramns and residuals
estimated_params = np.zeros((dwi_size, params_size, niter))
resnormall = np.zeros([dwi.shape[0],niter])

print('Estimated_params shape: ', estimated_params.shape)
Parameters = ['phi', 'theta', 'd_par', 'k_per', 'S0']
print('Parameters to be estimated: ', Parameters)
print('')
print('FITTING PARAMETERS ...                              ')
print('')


# Perform model fitting using least squares optimization
def objective_function(params):
    signal_pred = Zeppelin(params, g, bval)
    signal_pred = signal_pred.reshape(-1)
    residuals = S_voxel - signal_pred  
    return residuals

##################### Fit all slices ###########################################
estimated_params = np.zeros((dwi_size, params_size,niter))
residuals = np.zeros((dwi_size, bval.shape[0],niter))

start_time = time.time()

for ii in range(0,niter):
    print('niter ', ii , '/', niter)
    for pixel in range(0,dwi_size,1):
        S_voxel = dwi[pixel,:]
        S_b0 = max(np.max(S_voxel[bval < 0.1]), 0)
        x0 = np.random.rand(len(ub)) * ub
        x0[4] = S_b0
        
        result = least_squares(objective_function, x0, bounds=(lb,ub))
        estimated_params[pixel,:,ii] = result.x
        resnormall[pixel,ii] =  np.sum(result.fun**2)
        
            
end_time = time.time()

print('Resulting parameters from the fitting: ', estimated_params.shape)
print('') 
print(f'Execution time for dataset 10{image_idx}_DWI.nii with num iter {niter}: ', end_time - start_time, ' seconds')
print('')
print('... Done!')
print('')


# Find the indices of the minimum values along the second dimension of resnormall
I = np.argmin(resnormall, axis=1)

xall_ = np.zeros_like(estimated_params[:, :, 0])

# Iterate over voxels and fill xall_ using the indices I
for vox in range(estimated_params.shape[0]):
    xall_[vox, :] = estimated_params[vox, :,  I[vox]]
   
# unvec xall_residuals
xall_res = resnormall.T
xall_unvec_res = unvec(np.squeeze(xall_res), msk)

# unvec xall_parameters
xall_ = xall_.T
xall_unvec = unvec(np.squeeze(xall_), msk)

xall_unvec_ = xall_unvec  # to save the paramters with k_per always


####### VISUALIZE RESULTS FROM FITTING WITH LS. This will be my ground truth parameters #######

if vis ==1:
    num = 70   # specify the slice you want to visualize
    
    l1 = xall_unvec[:,:,num,2]  # axial diffusivity
    k = xall_unvec[:,:,num,3]   # kper
    l2 = l1*k                   # radial diffusivity
    xall_unvec[:,:,num,3] = l2
    
    mask = msk
    
    d_par_nc = l1 * [mask[:,:,num] == 1]
    d_per_nc = l2 * [mask[:,:,num] == 1]
    S0_nc = xall_unvec[:,:,num,4]*[mask[:,:,num] == 1]
    
    
    #### CALCULATE FA and MD ####
    axial_diffusivity= d_par_nc[0,:,:]
    radial_diffusivity = d_per_nc[0,:,:]
    
    # Define the diffusion tensor
    diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
    diffusion_tensor[..., 0] = radial_diffusivity
    diffusion_tensor[..., 1] = radial_diffusivity
    diffusion_tensor[..., 2] = axial_diffusivity
    
    FA_nc = fractional_anisotropy(diffusion_tensor)
    MD_nc =  (d_par_nc + d_per_nc + d_per_nc) / 3
    
    
    #### CALCULATE ANGLES IN CARTESIAN COORDINATES ####
    height = xall_unvec.shape[0]
    width = xall_unvec.shape[1]
    
    theta = xall_unvec[:,:,num,0] 
    phi = xall_unvec[:,:,num,1]
    
    # From spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Take the absolute values of the Cartesian coordinates
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z)
    
    # Create an RGB map with red, blue, and green
    rgb_map_pred = np.zeros((height,width, 3), dtype=np.uint8)
    rgb_map_pred[:, :, 0] = x_abs * 255   # Red channel
    rgb_map_pred[:, :, 1] = y_abs * 255   # Green channel
    rgb_map_pred[:, :, 2] = z_abs * 255   # Blue channel
    
    rgb_map_pred[:,:,0] = np.where(msk[:,:,num] == 1, rgb_map_pred[:,:,0], np.nan)
    rgb_map_pred[:,:,1] = np.where(msk[:,:,num] == 1, rgb_map_pred[:,:,1], np.nan)
    rgb_map_pred[:,:,2] = np.where(msk[:,:,num] == 1, rgb_map_pred[:,:,2], np.nan)
    
    rgb_map_pred2 = np.zeros((height,width, 3), dtype=np.uint8)
    rgb_map_pred2[:, :, 0] = rgb_map_pred[:, :, 0] * FA_nc   # Red channel
    rgb_map_pred2[:, :, 1] = rgb_map_pred[:, :, 1] * FA_nc   # Green channel
    rgb_map_pred2[:, :, 2] = rgb_map_pred[:, :, 2] * FA_nc   # Blue channel
    
    plt.imshow(rgb_map_pred2, cmap='jet')
    plt.title('rgb_map * FA map', fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()   
    
    plt.figure(figsize=(14, 9), facecolor='black')
    plt.suptitle(f'Ground truth parameters 10{image_idx}_DWI.nii', fontsize=25, color='white')
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(S0_nc[0,:,:]), cmap='gray')
    plt.title('S0', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(d_par_nc[0,:,:]), cmap='gray')
    plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(d_per_nc[0,:,:]), cmap='gray')
    plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 4)
    plt.imshow(np.rot90(FA_nc), cmap='gray')
    plt.title('FA map', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.rot90(MD_nc[0,:,:]), cmap='gray' )
    plt.title('MD [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.rot90(rgb_map_pred2), cmap='jet')
    plt.title('rgb_map principal direction º', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.show()


## Save results.

print('... Saving results ground truth parameters')

output_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx

## save predicted params as a nii flie
file_path = output_file_name + '_params.nii'
print('Output path: ', file_path)
nii_img = nib.Nifti1Image(xall_unvec_, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_img, file_path)

print('nii params saved!')
print('')


## save residuals as  as a nii flie
file_path = output_file_name + '_res.nii'
nii_res = nib.Nifti1Image(xall_unvec_res, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_res, file_path)

print('nii residuals saved!')

# File path
file_path = output_file_name + '_INFO.txt'

with open(file_path, 'w') as file:
    file.write('idx of image to fit: 10' + str(image_idx) + '_DWI.nii \n')
    file.write('Image shape: ' + str(dwi.shape)+ '\n')
    file.write('B-matrix shape: ' + str(b_matrix.shape)+ '\n')
    file.write('g (bvecs) shape: ' + str(g.shape)+ '\n')
    file.write('bvals shape: ' + str(bval.shape)+ '\n' + '\n')

    file.write('FITTING PRODEDURE INFO'+ '\n')
    file.write('Number of iterations to avoid local minima: ' + str(niter)+ '\n')
    file.write('Lower bounds: ' + str(lb)+ '\n')
    file.write('Upper bounds: ' + str(ub)+ '\n'  + '\n')
    
    file.write('Estimated_ params shape: ' + str(estimated_params.shape) + '\n')
    file.write('Parameters to be estimated: ' + str(Parameters) + '\n' + '\n')
    
    file.write('FITTING ... ' + '\n')
    file.write('Execution time dataset is ' + str(end_time - start_time) + ' sec'  + '\n' )

print('')
print("INFO has been saved to the file.")

my_file = open('{}_mask.bin'.format(output_file_name),'wb')
pk.dump(msk,my_file,pk.HIGHEST_PROTOCOL)
my_file.close()
print('')
print('mask saved!')

print('')  
print('*************************************************************************************')
print('                       PART 2: GENERATE GROUND TRUTH SIGNALS                         ')
print('*************************************************************************************')
print('')


print('')  
print('********************* WITH GRADIENT NON LINEARITIES INCLUDED ***********************')
print('')

path_xall_unvec = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx + '_params.nii'
xall_unvec, affine = load_nifti(path_xall_unvec)
print('xall_unvec shape:', xall_unvec.shape)

# path to gradient deviation correction file
print    
print('Loading grad_dev_file ...                  ')
print('')

path_grad_dev = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\10' + image_idx + '_grad_dev.nii.gz'
grad_dev, affine = load_nifti(path_grad_dev)
print('grad_dev shape:', grad_dev.shape)
print('')

print('Extracting L matrix elements ...                  ')
print('')
print('SIMULATING SIGNAL...')

# store the corresponding L elements in the grad_dev variable
Lxx = grad_dev[:, :, :, 0]
Lxy = grad_dev[:, :, :, 1]
Lxz = grad_dev[:, :, :, 2]
Lyy = grad_dev[:, :, :, 3]
Lyx = grad_dev[:, :, :, 4]
Lyz = grad_dev[:, :, :, 5]
Lzz = grad_dev[:, :, :, 6]
Lzx = grad_dev[:, :, :, 7]
Lzy = grad_dev[:, :, :, 8] 


x_size = xall_unvec.shape[0]
y_size = xall_unvec.shape[1]
z_size = xall_unvec.shape[2]

simulated_dwi_GNL = np.zeros((x_size,y_size,z_size,bval.shape[0]))

start_time = time.time()
for z in range(0,z_size,1):
    print('num slice ', z , '/', z_size)
    for x in range(0,x_size,1):
        for y in range(0,y_size,1):
            if not np.isnan(xall_unvec[x, y, z, 0]):
                
                # construct L matrix
                L = np.array([[Lxx[x,y,z], Lxy[x,y,z], Lxz[x,y,z]],
                                  [Lyx[x,y,z], Lyy[x,y,z], Lyz[x,y,z]],
                                  [Lzx[x,y,z], Lxy[x,y,z], Lzz[x,y,z]]])
                 
                # Apply gradient non-linearities correction
                I = np.eye(3)                          # identity matrix               
                v = np.dot(g, (I + L))
                n = np.sqrt(np.diag(np.dot(v, v.T)))
                
                new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
                new_bval = n ** 2 * bval
                new_bvec[new_bval == 0, :] = 0
                bval_corrected = new_bval 
                g_effective = new_bvec
                
                signal = Zeppelin(xall_unvec[x,y,z,:], g_effective, bval_corrected)
                signal = signal.reshape(-1)
                
                simulated_dwi_GNL[x,y,z,:] = signal
                
                
end_time = time.time()

print('')
print('Time to simulate signal from predicted parameters with GNL included: ', end_time - start_time, 'seconds')
print('')   

if vis == 1:
    
    ### check how the model did doing an image substraction and plotting a histogram of the differences
    # Read the image
    file_path_nii = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\10' + image_idx + '_DWI.nii'
    real_dwi, affine = load_nifti(file_path_nii)
    
    num = 70
    num_bval = 50
    
    plt.imshow(simulated_dwi_GNL[:,:,num,num_bval], cmap = 'gray')
    plt.title(f'simulated_dwi_GNL. Slice {num}, bval {num_bval}')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    plt.imshow(dwi_[:,:,num,num_bval], cmap = 'gray')
    plt.title(f'REAL dwi_. Slice {num}, bval {num_bval}')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    plt.imshow(np.abs(real_dwi[:,:,num,num_bval] - simulated_dwi_GNL[:,:,num,num_bval]), cmap = 'gray')
    plt.title(f'REAL dwi_ - simulated_dwi_GNL. Slice {num}, bval {num_bval}')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    # plot hist of differences
    signals_diff = real_dwi[:,:,num,num_bval] - simulated_dwi_GNL[:,:,num,num_bval]
    signals_diff = signals_diff[msk[:,:,num] == True]
    
    plt.hist(signals_diff)
    plt.title(f'REAL - simulated with GNL. Slice {num}, bval {num_bval}')
    plt.show()
    

## Save results.

print('... Saving results ground truth signals with GNL included')
output_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx

## save predicted params as a nii flie
file_path = output_file_name + '_dwi_GNL.nii'
print('Output path: ', file_path)
nii_img = nib.Nifti1Image(simulated_dwi_GNL, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_img, file_path)

print('nii dwi_GNL saved!')
print('')


print('')  
print('************************************************************************************************')
print('**  PART 3: ADD NOISE TO GT SIMULATED SIGNALS. Create dataset for training to get predictions **')
print('************************************************************************************************')
print('')


path_simulated_dwi_GNL = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx + '_dwi_GNL.nii'
simulated_dwi_GNL, affine = load_nifti(path_simulated_dwi_GNL)
print('simulated_dwi_GNL shape:', simulated_dwi_GNL.shape)

path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx + '_mask.bin'
with open(path_msk, 'rb') as file:
    msk = pk.load(file)  

# three types of noise are implemented: gauss, rician and non central chi squared

def gaussian_noise(image, snr):
    s0 = np.mean(image) # mean signal intensity of the image
    sigma_noise = s0 / snr
    print('sigma_noise: ', sigma_noise)
    print('mean signal intensity: ', s0)

    gaussian_noise = np.random.normal(scale=sigma_noise, size=image.shape) # Generate Gaussian noise
    noisy_image = image + gaussian_noise
    return noisy_image


def rician_noise(image, snr):
    s0 = np.mean(image) # mean signal intensity of the image
    sigma_noise = s0 / snr
    print('sigma_noise: ', sigma_noise)
    print('mean signal intensity: ', s0)
    
    gaussian_noise = np.random.normal(scale=np.sqrt(sigma_noise / 2), size=image.shape) # Gaussian noise
    rayleigh_noise = np.random.rayleigh(scale=np.sqrt(sigma_noise / (4 - np.pi)), size=image.shape) # Rayleigh-distributed noise
    rician_noise = np.sqrt(gaussian_noise**2 + rayleigh_noise**2) # combine them
    noisy_image = image + rician_noise
    return noisy_image


def noncentral_chisquare_noise(image, snr):
    s0 = np.mean(image) # mean signal intensity of the image
    sigma_noise = s0 / snr
    print('sigma_noise: ', sigma_noise)
    print('mean signal intensity: ', s0)
    
    gaussian_noise = np.random.normal(scale=np.sqrt(sigma_noise), size=image.shape) #  Gaussian noise
    noncentral_chisquare_noise = np.random.noncentral_chisquare(df=2, nonc=10, size=image.shape) # non-central chi-square noise
    noncentral_chisquare_noise = np.sqrt(gaussian_noise**2 + noncentral_chisquare_noise) # combine
    noisy_image = image + noncentral_chisquare_noise
    return noisy_image



## ADD NOISE

# flatten simulated dwi with correction
simulated_dwi_GNL_flat = vec(simulated_dwi_GNL,msk)
simulated_dwi_GNL_flat = simulated_dwi_GNL_flat[0].T
print('simulated_dwi_GNL_flat: ', simulated_dwi_GNL_flat.shape)

num = 70   # num slice to plot
num_bval = 1  # num bval to plot
noisetype = 'gaussian'
# Set the SNR
target_snr = 70

if noisetype == 'noncentral_chisquare':
    noisy_image = noncentral_chisquare_noise(simulated_dwi_GNL_flat, target_snr)
if noisetype == 'gaussian':
    noisy_image = gaussian_noise(simulated_dwi_GNL_flat, target_snr)
if noisetype == 'rician':
    noisy_image = rician_noise(simulated_dwi_GNL_flat, target_snr)
         
# unvec the image to plot it and see the noise
signal_unvec_noised_GNL = unvec(noisy_image.T, msk)
signal_unvec_noised_GNL = np.nan_to_num(signal_unvec_noised_GNL, nan=0)

plt.figure(figsize=(20, 6))
plt.suptitle(f'Dataset: 10{image_idx}. Slice [{num}], bval = {num_bval}. Noise: {noisetype}', fontsize=20)

plt.subplot(1, 3, 1)
plt.imshow(np.rot90(simulated_dwi_GNL[:,:,num,num_bval]), cmap = 'gray')
plt.title('simulated_dwi_GNL')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(np.rot90(signal_unvec_noised_GNL[:,:,num,num_bval]), cmap = 'gray')
plt.title('signal_noised_GNL')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(np.rot90(np.abs(signal_unvec_noised_GNL[:,:,num,num_bval]- simulated_dwi_GNL[:,:,num,num_bval])), cmap = 'gray')
plt.title(' |signal_noised_GNL- simulated_dwi_GNL|')
plt.axis('off')
plt.colorbar()
plt.show()

## Save results.

print('... Saving results signals + noise with gradient deviations')
output_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx

## save predicted params as a nii flie
file_path = output_file_name + '_dwi_noised_GNL.nii'
print('Output path: ', file_path)
nii_img = nib.Nifti1Image(signal_unvec_noised_GNL, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_img, file_path)

print('nii _dwi_noised_GNL saved!')
print('')


