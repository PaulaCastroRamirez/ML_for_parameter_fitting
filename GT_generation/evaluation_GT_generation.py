# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:48:22 2023

@author: pcastror

This code evaluates the results from the parameter fitting. Required input:
    
        - The correction you want to be dispalyed. Possible numbers 0,1,2.
               - 0 if i want no correction only 
               - 1 if i want correction only 
               - 2 if i want both compared
               - 3 if i want simulations
               - 4 if i want differences in angles only
               - 5 if i want to visualize the distributions
               
        - The path to the images:
            * without correction, the images with correction, and their corresponding masks
        
        - The slice numebr to be displayed
        
"""


# import necessary pakages

from dipy.io.image import load_nifti
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pk
from dipy.reconst.dti import fractional_anisotropy
import matplotlib.cm as cm
from utilities import Zeppelin, bval_bvec_from_b_Matrix

# what do you want to visualize
correction = 2


# specify the slice to plot
num = 70
bvalue= 0

# determine the path to the images to be evaluated
file_path_nii_nc  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\niter5_correct2_params.nii'
file_path_nii_c   = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\niter5_corrected2_grad_params.nii'
file_path_mask_nc = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\estimated_params_no_correction_try1_ALL_100307_DWI_mask.bin'
file_path_mask_c  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\correct_estimations_GT\estimated_params_no_correction_try1_ALL_100307_DWI_mask.bin'


########################### EVALUATION OF RESULTS ###############################

# read no correction
estimated_params_nc, affine_estimated_params = load_nifti(file_path_nii_nc)
# read correction
estimated_params_c, affine_estimated_params = load_nifti(file_path_nii_c)

# read mask no correction
with open(file_path_mask_nc, 'rb') as file:
    file_name = os.path.basename(file.name)
    mask_binary_nc = pk.load(file)  
    
# read mask correction
with open(file_path_mask_c, 'rb') as file:
    file_name = os.path.basename(file.name)
    mask_binary_c = pk.load(file)  

path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_100307_mask.bin'
with open(path_msk, 'rb') as file:
    mask_binary_nc = pk.load(file) 

mask_binary_c = mask_binary_nc

ang1_nc = estimated_params_nc [:,:,num,0]
ang2_nc = estimated_params_nc [:,:,num,1]
d_par_nc = estimated_params_nc [:,:,num,2]
k_nc = estimated_params_nc [:,:,num,3]
d_per_nc = d_par_nc * k_nc
S0_nc = estimated_params_nc [:,:,num,4]


ang1_c = estimated_params_c [:,:,num,0]
ang2_c = estimated_params_c [:,:,num,1]
d_par_c = estimated_params_c [:,:,num,2]
k_c = estimated_params_c [:,:,num,3]
d_per_c =  d_par_c * k_c
S0_c = estimated_params_c [:,:,num,4]


if correction == 0 or correction == 2:
    
    # Plot the binary mask using Matplotlib
    plt.imshow(mask_binary_nc[:,:,num], cmap='gray')
    plt.title('Binary Mask no correction')
    plt.show()
     
    #### CALCULATE FA and MD ####
    # # Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
    axial_diffusivity= d_par_nc
    radial_diffusivity = d_per_nc
    
    # Define the diffusion tensor using the voxel-wise diffusivity data
    diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
    diffusion_tensor[..., 0] = radial_diffusivity
    diffusion_tensor[..., 1] = radial_diffusivity
    diffusion_tensor[..., 2] = axial_diffusivity
    

    FA_nc = fractional_anisotropy(diffusion_tensor)
    MD_nc =  (d_par_nc + d_per_nc + d_per_nc) / 3
    
    # Define the size of your image
    height = ang1_nc.shape[0]
    width = ang1_nc.shape[1]
    
    theta = ang1_nc 
    phi = ang2_nc
    
    # From spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    x = np.where(mask_binary_nc[:,:,num] == 1, x, np.nan)
    y = np.where(mask_binary_nc[:,:,num] == 1, y, np.nan)
    z = np.where(mask_binary_nc[:,:,num] == 1, z, np.nan)
    
    # Take the absolute values of the Cartesian coordinates
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z)
    
    # Create an RGB map with red, blue, and green
    rgb_map_nc_ = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_map_nc_[:, :, 0] = x_abs * 255   # Red channel
    rgb_map_nc_[:, :, 1] = y_abs * 255   # Green channel
    rgb_map_nc_[:, :, 2] = z_abs * 255   # Blue channel
    
    rgb_map_nc = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_map_nc[:, :, 0] = rgb_map_nc_[:, :, 0] * FA_nc   
    rgb_map_nc[:, :, 1] = rgb_map_nc_[:, :, 1] * FA_nc 
    rgb_map_nc[:, :, 2] = rgb_map_nc_[:, :, 2] * FA_nc 
    
    # Display the RGB map
    plt.imshow(rgb_map_nc, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title('Angles no correction')
    plt.show()
    
    
    plt.figure(figsize=(14, 9), facecolor='black')
    plt.suptitle('Results from fitting with LS no correction', fontsize=25, color='white')

    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(S0_nc), cmap='gray')
    plt.title('S0', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(d_par_nc), cmap='gray')
    plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(d_per_nc), cmap='gray')
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
    plt.imshow(np.rot90(MD_nc), cmap='gray')
    plt.title('MD [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.rot90(rgb_map_nc), cmap='jet')
    plt.title('rgb_map principal direction', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.show()
    
    
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.suptitle('Results from fitting no correction', fontsize=25, color='white')

    plt.subplot(2, 2, 1)
    plt.imshow(np.rot90(S0_nc), cmap='gray')
    plt.title('S0', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 2, 2)
    plt.imshow(np.rot90(d_par_nc), cmap='gray')
    plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 2, 3)
    plt.imshow(np.rot90(d_per_nc), cmap='gray')
    plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 2, 4)
    plt.imshow(np.rot90(FA_nc), cmap='gray')
    plt.title('FA map', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.tight_layout()
    plt.show()
    

    print('')
    print('Summary statistics for data with no correction')
    
    # Calculate statistics
    S0_mean = np.mean(S0_nc[mask_binary_nc[:,:,num]== 1])
    S0_min = np.min(S0_nc[mask_binary_nc[:,:,num]== 1])
    S0_max = np.max(S0_nc[mask_binary_nc[:,:,num]== 1])
    
    dpar_mean = np.mean(d_par_nc[mask_binary_nc[:,:,num]== 1])
    dpar_min = np.min(d_par_nc[mask_binary_nc[:,:,num]== 1])
    dpar_max = np.max(d_par_nc[mask_binary_nc[:,:,num]== 1])
    
    dper_mean = np.mean(d_per_nc[mask_binary_nc[:,:,num]== 1])
    dper_min = np.min(d_per_nc[mask_binary_nc[:,:,num]== 1])
    dper_max = np.max(d_per_nc[mask_binary_nc[:,:,num]== 1])
    
    ang2_mean = np.mean(ang2_nc[mask_binary_nc[:,:,num]== 1])
    ang2_min = np.min(ang2_nc[mask_binary_nc[:,:,num]== 1])
    ang2_max = np.max(ang2_nc[mask_binary_nc[:,:,num]== 1])
    
    ang1_mean = np.mean(ang1_nc[mask_binary_nc[:,:,num]== 1])
    ang1_min = np.min(ang1_nc[mask_binary_nc[:,:,num]== 1])
    ang1_max = np.max(ang1_nc[mask_binary_nc[:,:,num]== 1])
    
    FA_mean = np.mean(FA_nc[mask_binary_nc[:,:,num]== 1])
    FA_min = np.min(FA_nc[mask_binary_nc[:,:,num]== 1])
    FA_max = np.max(FA_nc[mask_binary_nc[:,:,num]== 1])
    
    # Create a table to display the statistics
    statistics_table = {
        '': ['S0', 'dper', 'dpar', 'ang1', 'ang2', 'FA'],
        'Mean': [S0_mean, dper_mean, dpar_mean, ang1_mean, ang2_mean, FA_mean],
        'Min': [S0_min, dper_min, dpar_min, ang1_min, ang2_min, FA_min],
        'Max': [S0_max, dper_max, dpar_max, ang1_max, ang2_max, FA_max]
    }
    
        
    df_stats = pd.DataFrame(statistics_table)
    print(df_stats)
    
    
    # plot histograms 
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    fig.suptitle('Plot histograms no correction', fontsize=25)

    S0_nc_hist = S0_nc[mask_binary_nc[:,:,num] == 1]
    axes[0, 0].hist(S0_nc_hist, color='blue', label='estimation')
    axes[0, 0].set_title('S0')
    axes[0, 0].set_xlabel('S0')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    d_par_nc_hist = d_par_nc[mask_binary_nc[:,:,num] == 1]
    axes[0, 1].hist(d_par_nc_hist, color='blue', label='estimation')
    axes[0, 1].set_title('d_||/AD [μm^2/ms]')
    axes[0, 1].set_xlabel('d_|| [μm^2/ms]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    d_per_nc_hist = d_per_nc[mask_binary_nc[:,:,num] == 1]
    axes[0, 2].hist(d_per_nc_hist, color='blue', label='estimation')
    axes[0, 2].set_title('d_⊥/RD [μm^2/ms]')
    axes[0, 2].set_xlabel('d_⊥ [μm^2/ms]')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    ang1_nc_hist = ang1_nc[mask_binary_nc[:,:,num] == 1]
    axes[1, 0].hist(ang1_nc_hist, color='blue', label='estimation')
    axes[1, 0].set_title('ang1')
    axes[1, 0].set_xlabel('ang1 [rad]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    ang2_nc_hist = ang2_nc[mask_binary_nc[:,:,num] == 1]
    axes[1, 1].hist(ang2_nc_hist, color='blue', label='estimation')
    axes[1, 1].set_title('ang2')
    axes[1, 1].set_xlabel('ang2 [rad]')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    FA_nc_hist = FA_nc[mask_binary_nc[:,:,num] == 1]
    axes[1, 2].hist(FA_nc_hist, color='blue', label='estimation')
    axes[1, 2].set_title('FA map')
    axes[1, 2].set_xlabel('FA')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()
    

    
if correction == 1 or correction == 2:
    # Plot the binary mask using Matplotlib
    plt.imshow(mask_binary_c[:,:,num], cmap='gray')
    plt.title('Binary Mask correction')
    plt.show()

    #### CALCULATE FA and MD ####

    # # Assuming you have axial_diffusivity and radial_diffusivity voxel-wise data
    axial_diffusivity= d_par_c
    radial_diffusivity = d_per_c
    
    # Define the diffusion tensor using the voxel-wise diffusivity data
    diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
    diffusion_tensor[..., 0] = radial_diffusivity
    diffusion_tensor[..., 1] = radial_diffusivity
    diffusion_tensor[..., 2] = axial_diffusivity   
    
    FA_c = fractional_anisotropy(diffusion_tensor)
    MD_c = (d_par_c + d_per_c + d_per_c)/3 

    
    #### CALCULATE rgb maps ####

    # Define the size of your image
    height = ang1_c.shape[0]
    width = ang1_c.shape[1]
    
    theta = ang1_c 
    phi = ang2_c
    
    # From spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    x = np.where(mask_binary_c[:,:,num] == 1, x, np.nan)
    y = np.where(mask_binary_c[:,:,num] == 1, y, np.nan)
    z = np.where(mask_binary_c[:,:,num] == 1, z, np.nan)
    
    
    # Take the absolute values of the Cartesian coordinates
    x_abs = np.abs(x)
    y_abs = np.abs(y)
    z_abs = np.abs(z)
    
    # Create an RGB map with red, blue, and green
    rgb_map_c_ = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_map_c_[:, :, 0] = x_abs * 255   # Red channel
    rgb_map_c_[:, :, 1] = y_abs * 255   # Green channel
    rgb_map_c_[:, :, 2] = z_abs * 255   # Blue channel
    
    # Create an RGB map with red, blue, and green
    rgb_map_c = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_map_c[:, :, 0] = rgb_map_c_[:, :, 0] * FA_c
    rgb_map_c[:, :, 1] = rgb_map_c_[:, :, 1] * FA_c
    rgb_map_c[:, :, 2] = rgb_map_c_[:, :, 2] * FA_c
    
    # Display the RGB map
    plt.imshow(rgb_map_c, cmap='jet')
    plt.axis('off')
    plt.title('Angles with correction')
    plt.colorbar()
    plt.show()
     

    # plot

    plt.figure(figsize=(14, 9), facecolor='black')
    plt.suptitle('Results from fitting with correction', fontsize=25, color='white')

    plt.subplot(2, 3, 1)
    plt.imshow(S0_c, cmap='gray')
    plt.title('S0', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 3, 2)
    plt.imshow(d_par_c, cmap='gray')
    plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 3, 3)
    plt.imshow(d_per_c, cmap='gray')
    plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 3, 4)
    plt.imshow(FA_c, cmap='gray')
    plt.title('FA map', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 3, 5)
    plt.imshow(MD_c, cmap='gray')
    plt.title('MD [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')
    
    plt.subplot(2, 3, 6)
    plt.imshow(rgb_map_c, cmap='jet')
    plt.title('rgb_map principal direction', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.show()
    
    
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.suptitle('Results from fitting with correction', fontsize=25, color='white')

    plt.subplot(2, 2, 1)
    plt.imshow(S0_c, cmap='gray')
    plt.title('S0', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 2, 2)
    plt.imshow(d_par_c, cmap='gray')
    plt.title('AD/d_par [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 2, 3)
    plt.imshow(d_per_c, cmap='gray')
    plt.title('RD/d_per [μm^2/ms]', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.subplot(2, 2, 4)
    plt.imshow(FA_c, cmap='gray')
    plt.title('FA map', color='white',fontsize=15)
    plt.axis('off')
    color_bar = plt.colorbar()                          
    cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
    plt.setp(cbytick_obj, color='white')

    plt.tight_layout()
    plt.show()
    
    print('')
    print('Summary statistics for data with correction')

    # Calculate statistics
    S0_mean = np.mean(S0_c[mask_binary_c[:,:,num] == 1])
    S0_min = np.min(S0_c[mask_binary_c[:,:,num] == 1])
    S0_max = np.max(S0_c[mask_binary_c[:,:,num] == 1])

    dpar_mean = np.mean(d_par_c[mask_binary_c[:,:,num] == 1])
    dpar_min = np.min(d_par_c[mask_binary_c[:,:,num] == 1])
    dpar_max = np.max(d_par_c[mask_binary_c[:,:,num] == 1])

    dper_mean = np.mean(d_per_c[mask_binary_c[:,:,num] == 1])
    dper_min = np.min(d_per_c[mask_binary_c[:,:,num] == 1])
    dper_max = np.max(d_per_c[mask_binary_c[:,:,num] == 1])

    ang2_mean = np.mean(ang2_c[mask_binary_c[:,:,num] == 1])
    ang2_min = np.min(ang2_c[mask_binary_c[:,:,num] == 1])
    ang2_max = np.max(ang2_c[mask_binary_c[:,:,num] == 1])

    ang1_mean = np.mean(ang1_c[mask_binary_c[:,:,num] == 1])
    ang1_min = np.min(ang1_c[mask_binary_c[:,:,num] == 1])
    ang1_max = np.max(ang1_c[mask_binary_c[:,:,num] == 1])
    
    FA_mean = np.mean(FA_c[mask_binary_c[:,:,num] == 1])
    FA_min = np.min(FA_c[mask_binary_c[:,:,num] == 1])
    FA_max = np.max(FA_c[mask_binary_c[:,:,num] == 1])

    # Create a table to display the statistics
    statistics_table = {
        '': ['S0', 'dper', 'dpar', 'ang1', 'ang2', 'FA'],
        'Mean': [S0_mean, dper_mean, dpar_mean, ang1_mean, ang2_mean, FA_mean],
        'Min': [S0_min, dper_min, dpar_min, ang1_min, ang2_min, FA_min],
        'Max': [S0_max, dper_max, dpar_max, ang1_max, ang2_max, FA_max]
    }


    df_stats = pd.DataFrame(statistics_table)
    print(df_stats)
    print('')
    
    # plot histograms 
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    fig.suptitle('Plot histograms with correction', fontsize=25)

    S0_c_hist = S0_c[mask_binary_c[:,:,num] == 1]
    axes[0, 0].hist(S0_c_hist, color='blue', label='estimation')
    axes[0, 0].set_title('S0')
    axes[0, 0].set_xlabel('S0')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    d_par_c_hist = d_par_c[mask_binary_c[:,:,num] == 1]
    axes[0, 1].hist(d_par_c_hist, color='blue', label='estimation')
    axes[0, 1].set_title('d_||/AD [μm^2/ms]')
    axes[0, 1].set_xlabel('d_|| [μm^2/ms]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    d_per_c_hist = d_per_c[mask_binary_c[:,:,num] == 1]
    axes[0, 2].hist(d_per_c_hist, color='blue', label='estimation')
    axes[0, 2].set_title('d_⊥/RD [μm^2/ms]')
    axes[0, 2].set_xlabel('d_⊥ [μm^2/ms]')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    ang1_c_hist = ang1_c[mask_binary_c[:,:,num] == 1]
    axes[1, 0].hist(ang1_c_hist, color='blue', label='estimation')
    axes[1, 0].set_title('ang1')
    axes[1, 0].set_xlabel('ang1 [rad]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    ang2_c_hist = ang2_c[mask_binary_c[:,:,num] == 1]
    axes[1, 1].hist(ang2_c_hist, color='blue', label='estimation')
    axes[1, 1].set_title('ang2')
    axes[1, 1].set_xlabel('ang2 [rad]')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    FA_c_hist = FA_c[mask_binary_c[:,:,num] == 1]
    axes[1, 2].hist(FA_c_hist, color='blue', label='estimation')
    axes[1, 2].set_title('FA map')
    axes[1, 2].set_xlabel('FA')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()
    
    

if correction == 2:
    

    rgb_map_diff = np.empty((rgb_map_nc.shape[0], rgb_map_nc.shape[1]))
    rgb_map_diff3 = np.empty((rgb_map_nc.shape[0], rgb_map_nc.shape[1]))

    dot_product = np.empty((rgb_map_nc.shape[0], rgb_map_nc.shape[1]))
    for pix_x in range(0,rgb_map_nc.shape[0],1):
        for pix_y in range(0,rgb_map_nc.shape[1],1):
            vector1 = np.array(rgb_map_nc[pix_x,pix_y,:])
            vector1 = np.double(vector1)
            vector2 = np.array(rgb_map_c[pix_x,pix_y,:])
            vector2 = np.double(vector2)
            dot_product[pix_x,pix_y] = np.dot(vector1, vector2)
            #print(dot_product)
            rgb_map_diff[pix_x,pix_y] = np.arccos(dot_product[pix_x,pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
            rgb_map_diff3[pix_x,pix_y] = (dot_product[pix_x,pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    
    empty_rows = np.where(np.isnan(rgb_map_diff))
    # Replace empty vectors with NaN values
    rgb_map_diff[empty_rows] = 0
    rgb_map_diff = np.where(mask_binary_nc[:,:,num] == 1, rgb_map_diff, np.nan)
    
    cmap = plt.get_cmap("jet")  
    cmap.set_bad("black", alpha=1.0)  # Map NaN values to black

    # Plot the angle differences in radians
    plt.imshow(rgb_map_diff, cmap=cmap)
    plt.colorbar()
    plt.title("Angle Differences")
    plt.axis('off')
    plt.show()
    
    rgb_map_diff2 = 1 - rgb_map_diff3
    
    # Plot the angle differences in radians
    plt.imshow(rgb_map_diff2, cmap=cmap)
    plt.colorbar()
    plt.title("Angle Differences (1-|x*y|)")
    plt.axis('off')
    plt.show()
    
    
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(7, 15))
    fig.suptitle('No correction vs corrected', fontsize=16)
    
    # Plot data in each subplot
    im = axes[0, 0].imshow(S0_nc, cmap='gray')
    axes[0, 0].set_title('S0 no correction')
    axes[0, 0].axis('off')
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(S0_c, cmap='gray')
    axes[0, 1].set_title('S0 corrected')
    axes[0, 1].axis('off')
    fig.colorbar(im, ax=axes[0, 1])
    
    im = axes[1, 0].imshow(d_par_nc, cmap='gray')
    axes[1, 0].set_title('AD no correction [μm^2/ms]')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(d_par_c, cmap='gray')
    axes[1, 1].set_title('AD corrected [μm^2/ms]')
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1])
    
    im = axes[2, 0].imshow(d_per_nc, cmap='gray')
    axes[2, 0].set_title('RD no correction')
    axes[2, 0].axis('off')
    fig.colorbar(im, ax=axes[2, 0])
    
    im = axes[2, 1].imshow(d_per_c, cmap='gray')
    axes[2, 1].set_title('d_per_c')
    axes[2, 1].axis('off')
    fig.colorbar(im, ax=axes[2, 1])
    
    im = axes[3, 0].imshow(FA_nc, cmap='gray')
    axes[3, 0].set_title('FA map no correction')
    axes[3, 0].axis('off')
    fig.colorbar(im, ax=axes[3, 0])
    
    im = axes[3, 1].imshow(FA_c, cmap='gray')
    axes[3, 1].set_title('FA map correcred')
    axes[3, 1].axis('off')
    fig.colorbar(im, ax=axes[3, 1])
    
    im = axes[4, 0].imshow(MD_nc, cmap='gray')
    axes[4, 0].set_title('MD no correction')
    axes[4, 0].axis('off')
    fig.colorbar(im, ax=axes[4, 0])
    
    im = axes[4, 1].imshow(MD_c, cmap='gray')
    axes[4, 1].set_title('MD correcred')
    axes[4, 1].axis('off')
    fig.colorbar(im, ax=axes[4, 1])
    
    im = axes[5, 0].imshow(rgb_map_nc, cmap='jet')
    axes[5, 0].set_title('angles no correction')
    axes[5, 0].axis('off')
    fig.colorbar(im, ax=axes[5, 0])
    
    im = axes[5, 1].imshow(rgb_map_c, cmap='jet')
    axes[5, 1].set_title('angles with correction')
    axes[5, 1].axis('off')
    fig.colorbar(im, ax=axes[5, 1])
    
    # Adjust layout and display the plot
    plt.tight_layout()
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
    
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 15))
    fig.suptitle('No correction vs corrected vs absolute diff', fontsize=16)
    
    # Plot data in each subplot
    im = axes[0, 0].imshow(S0_nc, cmap='gray')
    axes[0, 0].set_title('S0 no correction')
    axes[0, 0].axis('off')
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(S0_c, cmap='gray')
    axes[0, 1].set_title('S0 corrected')
    axes[0, 1].axis('off')
    fig.colorbar(im, ax=axes[0, 1])
    
    im = axes[0, 2].imshow(np.abs(S0_nc-S0_c), cmap='gray')
    axes[0, 2].set_title('S0 diff')
    axes[0, 2].axis('off')
    fig.colorbar(im, ax=axes[0, 2])
    
    im = axes[1, 0].imshow(d_par_nc, cmap='gray')
    axes[1, 0].set_title('AD no correction')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(d_par_c, cmap='gray')
    axes[1, 1].set_title('AD corrected')
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].imshow(np.abs(d_par_nc - d_par_c), cmap='gray')
    axes[1, 2].set_title('AD diff')
    axes[1, 2].axis('off')
    fig.colorbar(im, ax=axes[1, 2])
    
    im = axes[2, 0].imshow(d_per_nc, cmap='gray')
    axes[2, 0].set_title('RD no correction')
    axes[2, 0].axis('off')
    fig.colorbar(im, ax=axes[2, 0])
    
    im = axes[2, 1].imshow(d_per_c, cmap='gray')
    axes[2, 1].set_title('d_per_c')
    axes[2, 1].axis('off')
    fig.colorbar(im, ax=axes[2, 1])
    
    im = axes[2, 2].imshow(np.abs(d_per_nc - d_per_c), cmap='gray')
    axes[2, 2].set_title('RD diff')
    axes[2, 2].axis('off')
    fig.colorbar(im, ax=axes[2, 2])
    
    im = axes[3, 0].imshow(FA_nc, cmap='gray')
    axes[3, 0].set_title('FA map no correction')
    axes[3, 0].axis('off')
    fig.colorbar(im, ax=axes[3, 0])
    
    im = axes[3, 1].imshow(FA_c, cmap='gray')
    axes[3, 1].set_title('FA map corrected')
    axes[3, 1].axis('off')
    fig.colorbar(im, ax=axes[3, 1])
    
    im = axes[3, 2].imshow(np.abs(FA_nc - FA_c), cmap='gray')
    axes[3, 2].set_title('FA diff')
    axes[3, 2].axis('off')
    fig.colorbar(im, ax=axes[3, 2])
    
    im = axes[4, 0].imshow(MD_nc, cmap='gray')
    axes[4, 0].set_title('MD no correction')
    axes[4, 0].axis('off')
    fig.colorbar(im, ax=axes[4, 0])
    
    im = axes[4, 1].imshow(MD_c, cmap='gray')
    axes[4, 1].set_title('MD corrected')
    axes[4, 1].axis('off')
    fig.colorbar(im, ax=axes[4, 1])
    
    im = axes[4, 2].imshow(np.abs(MD_nc - MD_c), cmap='gray')
    axes[4, 2].set_title('MD diff')
    axes[4, 2].axis('off')
    fig.colorbar(im, ax=axes[4, 2])
    
    im = axes[5, 0].imshow(rgb_map_nc, cmap='jet')
    axes[5, 0].set_title('angles no correction')
    axes[5, 0].axis('off')
    fig.colorbar(im, ax=axes[5, 0])
    
    im = axes[5, 1].imshow(rgb_map_c, cmap='jet')
    axes[5, 1].set_title('angles with correction')
    axes[5, 1].axis('off')
    fig.colorbar(im, ax=axes[5, 1])
    
    im = axes[5, 2].imshow(np.abs(rgb_map_diff), cmap=cmap)
    axes[5, 2].set_title('cosθ = 1 - |rgb_nc . rgb_c| [%]')
    axes[5, 2].axis('off')
    fig.colorbar(im, ax=axes[5, 2])
        
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 15))
    fig.suptitle('No correction vs corrected', fontsize=16)
    
    # Plot data in each subplot
    im = axes[0, 0].imshow(S0_nc, cmap='gray')
    axes[0, 0].set_title('S0 no correction')
    axes[0, 0].axis('off')
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(S0_c, cmap='gray')
    axes[0, 1].set_title('S0 corrected')
    axes[0, 1].axis('off')
    fig.colorbar(im, ax=axes[0, 1])
    
    im = axes[0, 2].imshow(S0_nc-S0_c, cmap='gray')
    axes[0, 2].set_title('S0 diff')
    axes[0, 2].axis('off')
    fig.colorbar(im, ax=axes[0, 2])
    
    im = axes[1, 0].imshow(d_par_nc, cmap='gray')
    axes[1, 0].set_title('AD no correction')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(d_par_c, cmap='gray')
    axes[1, 1].set_title('AD corrected')
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].imshow(d_par_nc - d_par_c, cmap='gray')
    axes[1, 2].set_title('AD diff')
    axes[1, 2].axis('off')
    fig.colorbar(im, ax=axes[1, 2])
    
    im = axes[2, 0].imshow(d_per_nc, cmap='gray')
    axes[2, 0].set_title('RD no correction')
    axes[2, 0].axis('off')
    fig.colorbar(im, ax=axes[2, 0])
    
    im = axes[2, 1].imshow(d_per_c, cmap='gray')
    axes[2, 1].set_title('d_per_c')
    axes[2, 1].axis('off')
    fig.colorbar(im, ax=axes[2, 1])
    
    im = axes[2, 2].imshow(d_per_nc - d_per_c, cmap='gray')
    axes[2, 2].set_title('RD diff')
    axes[2, 2].axis('off')
    fig.colorbar(im, ax=axes[2, 2])
    
    im = axes[3, 0].imshow(FA_nc, cmap='gray')
    axes[3, 0].set_title('FA map no correction')
    axes[3, 0].axis('off')
    fig.colorbar(im, ax=axes[3, 0])
    
    im = axes[3, 1].imshow(FA_c, cmap='gray')
    axes[3, 1].set_title('FA map corrected')
    axes[3, 1].axis('off')
    fig.colorbar(im, ax=axes[3, 1])
    
    im = axes[3, 2].imshow(FA_nc - FA_c, cmap='gray')
    axes[3, 2].set_title('FA diff')
    axes[3, 2].axis('off')
    fig.colorbar(im, ax=axes[3, 2])
    
    im = axes[4, 0].imshow(MD_nc, cmap='gray')
    axes[4, 0].set_title('MD no correction')
    axes[4, 0].axis('off')
    fig.colorbar(im, ax=axes[4, 0])
    
    im = axes[4, 1].imshow(MD_c, cmap='gray')
    axes[4, 1].set_title('MD corrected')
    axes[4, 1].axis('off')
    fig.colorbar(im, ax=axes[4, 1])
    
    im = axes[4, 2].imshow(MD_nc - MD_c, cmap='gray')
    axes[4, 2].set_title('MD diff')
    axes[4, 2].axis('off')
    fig.colorbar(im, ax=axes[4, 2])
    
    im = axes[5, 0].imshow(rgb_map_nc, cmap='jet')
    axes[5, 0].set_title('angles no correction')
    axes[5, 0].axis('off')
    fig.colorbar(im, ax=axes[5, 0])
    
    im = axes[5, 1].imshow(rgb_map_c, cmap='jet')
    axes[5, 1].set_title('angles with correction')
    axes[5, 1].axis('off')
    fig.colorbar(im, ax=axes[5, 1])
    
    im = axes[5, 2].imshow(rgb_map_diff, cmap=cmap)
    axes[5, 2].set_title('rgb_map diff')
    axes[5, 2].axis('off')
    fig.colorbar(im, ax=axes[5, 2])
        
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
        
    plt.figure(figsize=(24, 10))
    plt.suptitle('Plot difference', fontsize=27)
    
    plt.subplot(2, 3, 1)
    plt.imshow(S0_nc - S0_c, cmap='gray')
    plt.title('S0 diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(d_par_nc - d_par_c, cmap='gray')
    plt.title('AD/d_par diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(d_per_nc - d_per_c, cmap='gray')
    plt.title('RD/d_per diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(FA_nc - FA_c, cmap='gray')
    plt.title('FA diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(MD_nc - MD_c, cmap='gray')
    plt.title('MD diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(rgb_map_diff, cmap=cmap)
    plt.title('rgb_map diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(24, 10))
    plt.suptitle('Plot absolute difference', fontsize=27)
    
    plt.subplot(2, 3, 1)
    plt.imshow(np.abs(S0_nc - S0_c), cmap='gray')
    plt.title('S0 diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.abs(d_par_nc - d_par_c), cmap='gray')
    plt.title('AD/d_par diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(d_per_nc - d_per_c), cmap='gray')
    plt.title('RD/d_per diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(np.abs(FA_nc - FA_c), cmap='gray')
    plt.title('FA diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.abs(MD_nc - MD_c), cmap='gray')
    plt.title('MD diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(np.abs(rgb_map_diff), cmap=cmap)
    plt.title('rgb_map diff', fontsize=20)
    plt.colorbar()  # Add a colorbar for reference
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(32, 5))
    fig.suptitle('Plot difference', fontsize=25)
    
    # Plot data in each subplot
    im = axes[0].imshow((S0_nc - S0_c), cmap='gray')
    axes[0].set_title('S0 diff', fontsize=20)
    axes[0].axis('off')
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(d_par_nc - d_par_c, cmap='gray')
    axes[1].set_title('AD diff', fontsize=20)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(d_per_nc - d_per_c, cmap='gray')
    axes[2].set_title('RD diff', fontsize=20)
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2])

    im = axes[3].imshow(FA_nc - FA_c, cmap='gray')
    axes[3].set_title('FA diff', fontsize=20)
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3])

    im = axes[4].imshow(MD_nc - MD_c, cmap='gray')
    axes[4].set_title('MD diff', fontsize=20)
    axes[4].axis('off')
    fig.colorbar(im, ax=axes[4])

    im = axes[5].imshow(rgb_map_diff, cmap=cmap)
    axes[5].set_title('rgb_map diff', fontsize=20)
    axes[5].axis('off')
    fig.colorbar(im, ax=axes[5])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    fig.suptitle('Plot difference', fontsize=16)
    
    # Plot data in each subplot
    im = axes[0].imshow(FA_nc, cmap='gray')
    axes[0].set_title('FA map no correction')
    axes[0].axis('off')
    fig.colorbar(im, ax=axes[0])
    
    im = axes[1].imshow(FA_c, cmap='gray')
    axes[1].set_title('FA map with correction')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(FA_nc-FA_c, cmap='gray')
    axes[2].set_title('FA map difference')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    fig.suptitle('Plot difference rgb maps', fontsize=16)
    
    # Plot data in each subplot
    im = axes[0].imshow(rgb_map_nc, cmap='jet')
    axes[0].set_title('rgb map no correction')
    axes[0].axis('off')
    fig.colorbar(im, ax=axes[0])
    
    im = axes[1].imshow(rgb_map_c, cmap='jet')
    axes[1].set_title('rgb map with correction')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(np.abs(rgb_map_diff), cmap=cmap)
    axes[2].set_title('rgb map difference')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    

if correction == 3:


    print('')
    print('*********** GENERATE SIMULATIONS ************')
    print('')
    # Read the HCP images and load them
    folder_path = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP'
    file_list = os.listdir(folder_path)
    data_dictionary = {}
    nii_files = [file for file in file_list if file.endswith('.nii')]
    bvec_files = [file for file in file_list if file.endswith('.txt')]
    # Iterate over the NIfTI files and load all of them
    for i, file in enumerate(nii_files):
        file_path_nii = os.path.join(folder_path, file)
        file_path_bvec = os.path.join(folder_path,  bvec_files[i])
        img, affine = load_nifti(file_path_nii)    
        b_matrix = np.loadtxt(file_path_bvec)
        data_dictionary [i] = {"nii_file": img, "b_matrix": b_matrix}
    # choose only one dataset. for example the one in position 0
    num_image = 0
    S = data_dictionary[num_image]['nii_file']
    b_matrix = data_dictionary[num_image]['b_matrix']
    bval, g = bval_bvec_from_b_Matrix (b_matrix)


    print('estimated_params_nc shape: ', estimated_params_nc.shape)
    x_size = estimated_params_nc.shape[0]
    y_size = estimated_params_nc.shape[1]
    signal_sim_nc = np.zeros((estimated_params_nc.shape[0], estimated_params_nc.shape[1],  bval.shape[0]))
    print('mask_binary_nc: ', mask_binary_nc.shape)
    
    for x in range(0,x_size,1):
        for y in range(0,y_size,1):
            if (mask_binary_nc[x,y,num]):
                signal_pred = Zeppelin(estimated_params_nc[x,y,:], g, bval)
                signal_pred = signal_pred.reshape(-1)
                signal_sim_nc[x,y,:] = signal_pred
     
    plt.figure(figsize=(20,6))
    plt.suptitle(f'Results from signal simulation no correction. Slice {num}, bvalue {bval[bvalue]}', fontsize=16)
    
    plt.subplot(1, 3, 1)
    S_masked = S[:,:,num,bvalue]*mask_binary_nc[:,:,num]
    plt.imshow(S_masked, cmap='gray')
    plt.title('Original signal (S)')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(signal_sim_nc[:,:,bvalue], cmap='gray')
    plt.title('signal_sim_nc')
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(S_masked - signal_sim_nc[:,:,bvalue]), cmap='gray')
    plt.title('S - signal_sim_nc')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if correction == 4:
    
    #### with correction ####
    # Define the size of your image
    height = ang1_c.shape[0]
    width = ang1_c.shape[1]
    
    theta = ang1_c 
    phi = ang2_c
    
    # From spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    x = np.where(mask_binary_c[:,:,num] == 1, x, np.nan)
    y = np.where(mask_binary_c[:,:,num] == 1, y, np.nan)
    z = np.where(mask_binary_c[:,:,num] == 1, z, np.nan)
    
    
    # Take the absolute values of the Cartesian coordinates
    x_abs_c = np.abs(x)
    y_abs_c = np.abs(y)
    z_abs_c = np.abs(z)
    
    # Create an RGB map with red, blue, and green
    rgb_map_c = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_map_c[:, :, 0] = x_abs_c * 255   # Red channel
    rgb_map_c[:, :, 1] = y_abs_c * 255   # Green channel
    rgb_map_c[:, :, 2] = z_abs_c * 255   # Blue channel
    
    
    # Display the RGB map
    plt.imshow(rgb_map_c, cmap='jet')
    plt.axis('off')
    plt.title('Angles with correction')
    plt.colorbar()
    plt.show()
    
    
    #### without correction ####
    # Define the size of your image
    height = ang1_nc.shape[0]
    width = ang1_nc.shape[1]
    
    theta = ang1_nc 
    phi = ang2_nc
    
    # From spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    x = np.where(mask_binary_nc[:,:,num] == 1, x, np.nan)
    y = np.where(mask_binary_nc[:,:,num] == 1, y, np.nan)
    z = np.where(mask_binary_nc[:,:,num] == 1, z, np.nan)
    
    # Take the absolute values of the Cartesian coordinates
    x_abs_nc = np.abs(x)
    y_abs_nc = np.abs(y)
    z_abs_nc = np.abs(z)
    
    # Create an RGB map with red, blue, and green
    rgb_map_nc = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_map_nc[:, :, 0] = x_abs_nc * 255   # Red channel
    rgb_map_nc[:, :, 1] = y_abs_nc * 255   # Green channel
    rgb_map_nc[:, :, 2] = z_abs_nc * 255   # Blue channel
    
    
    # Display the RGB map
    plt.imshow(rgb_map_nc, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title('Angles no correction')
    plt.show()
    
    
    plt.figure(figsize=(20,6))
    plt.suptitle('diff x', fontsize=16)
    plt.subplot(1, 3, 1)
    plt.imshow(x_abs_nc, cmap='jet')
    plt.title('x no correction')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(x_abs_c, cmap='jet')
    plt.title('x correction')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(x_abs_nc-x_abs_c), cmap='jet')
    plt.title('x no correction - x correction')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(20,6))
    plt.suptitle('diff y', fontsize=16)
    plt.subplot(1, 3, 1)
    plt.imshow(y_abs_nc, cmap='jet')
    plt.title('y no correction')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(y_abs_c, cmap='jet')
    plt.title('y correction')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(y_abs_nc-y_abs_c), cmap='jet')
    plt.title('y no correction - y correction')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(20,6))
    plt.suptitle('diff z', fontsize=16)
    plt.subplot(1, 3, 1)
    plt.imshow(z_abs_nc, cmap='jet')
    plt.title('z no correction')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(z_abs_c, cmap='jet')
    plt.title('z correction')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(z_abs_nc-z_abs_c), cmap='jet')
    plt.title('z no correction - z correction')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
       
    result = np.empty((rgb_map_nc.shape[0], rgb_map_nc.shape[1]))
    dot_product = np.empty((rgb_map_nc.shape[0], rgb_map_nc.shape[1]))
    for pix_x in range(0,rgb_map_nc.shape[0],1):
        for pix_y in range(0,rgb_map_nc.shape[1],1):
            vector1 = np.array(rgb_map_nc[pix_x,pix_y,:])
            vector1 = np.double(vector1)
            vector2 = np.array(rgb_map_c[pix_x,pix_y,:])
            vector2 = np.double(vector2)
            dot_product[pix_x,pix_y] = np.dot(vector1, vector2)
            #print(dot_product)
            result[pix_x,pix_y] = np.arccos(dot_product[pix_x,pix_y] / ((np.linalg.norm(vector1) * np.linalg.norm(vector2))))
    
    empty_rows = np.where(np.isnan(result))
    # Replace empty vectors with NaN values
    result[empty_rows] = 0
    result = np.where(mask_binary_nc[:,:,num] == 1, result, np.nan)
    
    
    cmap = plt.get_cmap("jet")
    cmap.set_bad("black", alpha=1.0)  # Map NaN values to black


    # Plot the angle differences in radians
    plt.imshow(result, cmap=cmap)
    plt.colorbar()
    plt.title("Angle Differences")
    plt.axis('off')
    plt.show()
    
    
if correction == 5:
    
    print('')
    ### distribution of values
    distr_ang1_nc  = ang1_nc[mask_binary_nc[:,:,num] == 1]
    print(f'ang1 inside brain {distr_ang1_nc.size} / {ang1_nc.size}')
    plt.hist(distr_ang1_nc, bins=50, density=True, alpha=0.7, color='b')
    plt.title('polar (ang1) real distribution')
    plt.show()


    distr_ang2_nc  = ang2_nc[mask_binary_nc[:,:,num] == 1]
    print(f'ang2 inside brain {distr_ang2_nc.size} / {ang2_nc.size}')
    plt.hist(distr_ang2_nc, bins=50, density=True, alpha=0.7, color='b')
    plt.title('azimutal (ang2) real distribution')
    plt.show()

    distr_S0_nc  = S0_nc[mask_binary_nc[:,:,num] == 1]
    print(f'S0 inside brain {distr_S0_nc.size} / {S0_nc.size}')
    plt.hist(distr_S0_nc, bins=50, density=True, alpha=0.7, color='b')
    plt.title('S0 real distribution')
    plt.show()

    distr_AD_nc  = d_par_nc[mask_binary_nc[:,:,num] == 1]
    print(f'AD inside brain {distr_AD_nc.size} / {d_par_nc.size}')
    plt.hist(distr_AD_nc, bins=50, density=True, alpha=0.7, color='b')
    plt.title('AD real distribution')
    plt.show()

    distr_RD_nc  = d_per_nc[mask_binary_nc[:,:,num] == 1]
    print(f'RD inside brain {distr_RD_nc.size} / {d_per_nc.size}')
    plt.hist(distr_RD_nc, bins=50, density=True, alpha=0.7, color='b')
    plt.title('RD real distribution')
    plt.show()
    
    



