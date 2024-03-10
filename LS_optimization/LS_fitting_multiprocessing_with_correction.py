# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:27:46 2023

@author: pcastror

This code fits images from the HCP to the Zeppelin model using Least Squares Optimization. 
It adds gradient deviation corrections. 
Multiple initializations.

THis code saves:
    - estimated paramters from LS fitting as .nii file
    - residuals for each iteration performed as .nii file
    - INFO .txt file with some detaisn of the fitting procedure

OPTIONAL: optional visualization of results while running script. 
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


print('') 
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('Device used. using GPU (cuda) will significantly speed up the training:', device) # Device used

image_idx_list = ['0307', '0408', '1006', '1107']
print('')

vis = 1

for image_idx in image_idx_list:
    
    print(F'********** FITTING IMAGE 10{image_idx} WITH LEAST SQUARES, GNL CORRETION INCLUDED **********')
    print('')
    
    # Read the image
    file_path_nii = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx + '_dwi_noised_GNL.nii'
    file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\10' +  image_idx + '_DWI.txt'
    dwi, affine = load_nifti(file_path_nii)    
    b_matrix = np.loadtxt(file_path_bvec)
    
    # path to gradient deviation correction file
    path_grad_dev = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\10' + image_idx + '_grad_dev.nii.gz'
    grad_dev, affine = load_nifti(path_grad_dev)
    print('grad_dev shape:', grad_dev.shape)
    
    dwi_ = dwi
    grad_dev_ = grad_dev
    
    # extract bvals and bvecs from b matrix
    bval, g = bval_bvec_from_b_Matrix (b_matrix)
    bval=bval/1000
    
    # choose only first 108 bvalues because im using a tensor model
    bval = bval[:108]
    dwi = dwi[:,:,:,:108]
    g = g[:108,:]
    
    # path to mask file
    path_msk = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\HCP\ground_truths\GT_10' + image_idx + '_mask.bin'
    with open(path_msk, 'rb') as file:
        msk = pk.load(file)  
    
    num = 70
    plt.imshow(np.rot90(dwi_[:,:,num,0]), cmap='gray')
    plt.imshow(np.rot90(msk[:,:,num]), cmap='viridis', alpha=0.2)
    plt.title(f'Masked dwi dataset 10{image_idx}')
    plt.axis('off')
    plt.show()
    
    dwi = vec(dwi,msk)
    dwi = dwi[0]
    dwi = dwi.T

    grad_dev = vec(grad_dev,msk)
    grad_dev = grad_dev[0]
    grad_dev = grad_dev.T
    
    niter = 5
    
    print('')    
    print('Extracting L mtrix elements ...                  ')
    print('')
    
    # store the corresponding L elements in the grad_dev variable
    Lxx = grad_dev[:, 0]
    Lxy = grad_dev[:, 1]
    Lxz = grad_dev[:, 2]
    Lyy = grad_dev[:, 3]
    Lyx = grad_dev[:, 4]
    Lyz = grad_dev[:, 5]
    Lzz = grad_dev[:, 6]
    Lzx = grad_dev[:, 7]
    Lzy = grad_dev[:, 8] 
    
    print('')  
    print('*************************************************************************************')
    print('                                  LOAD DATA TO FIT                                   ')
    print('*************************************************************************************')
    print('')
    
    
    # choose lower and upper bounds
    lb = [0, 0, 0, 0, 0]
    ub = [np.pi, 2*np.pi, 3.2, 1, np.inf]
    
    
    print('Image shape: ', dwi.shape)
    print('B-matrix shape: ', b_matrix.shape)
    print('g (bvecs): ',g.shape)
    print('bvals: ', bval.shape)
    
    
    print('')
    print('*************************************************************************************')
    print('                                FTT THE SIGNAL FROM HCP                             ')
    print('*************************************************************************************')
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
    Parameters = ['theta', 'phi', 'd_par', 'k_per', 'S0']
    print('Parameters to be estimated: ', Parameters)
    
    
    print('')
    print('Fitting...                              ')
    print('')
    
    
    # Perform model fitting using least squares optimization.
    def objective_function(params):
        signal_pred = Zeppelin(params, g_effective, bval_corrected)
        signal_pred = signal_pred.reshape(-1)
        residuals = S_voxel - signal_pred  
        return residuals
    
    ##################### Fit all slices ###########################################
        
    start_time = time.time()
    
    for ii in range(0,niter):
        print('niter ', ii , '/', niter)
        for pixel in range(0,dwi_size,1):
            S_voxel = dwi[pixel,:]
            
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
            
            S_b0 = max(np.max(S_voxel[bval < 0.1]), 0)
            x0 = np.random.rand(len(ub)) * ub
            x0[4] = S_b0
            
            result = least_squares(objective_function, x0, bounds=(lb,ub))
            estimated_params[pixel,:,ii] = result.x
            resnormall[pixel,ii] =  np.sum(result.fun**2)
            
                
    end_time = time.time()
    
    print('Resulting parameters from the fitting: ', estimated_params.shape)
    print('') 
    print('Execution time for dataset with num iter {niter}:: ', end_time - start_time, ' seconds')
    print('')
    print('... Done!')
    print('')
    
    
    # Find the indices of the minimum values along the second dimension of resnormall
    I = np.argmin(resnormall, axis=1)
    
    xall_ = np.zeros_like(estimated_params[:, :, 0])
    
    # Iterate over voxels and fill xall_ using the indices I
    for vox in range(estimated_params.shape[0]):
        xall_[vox, :] = estimated_params[vox, :,  I[vox]]
    
    xall_ = xall_.T
    xall_unvec = unvec(np.squeeze(xall_),msk)
    
    xall_res = resnormall.T
    xall_unvec_res = unvec(np.squeeze(xall_res), msk)
    
    xall_unvec_ = xall_unvec  # to save the paramters with k_per always
    
    
    ####### VISUALIZE RESULTS FROM FITTING WITH LS #######
    
    if vis == 1:
        
        num = 90
        
        l1 = xall_unvec[:,:,num,2]
        k = xall_unvec[:,:,num,3] 
        l2 = l1*k
        xall_unvec[:,:,num,3] = l2
        
        mask = msk
        
        d_par = l1 * [mask[:,:,num] == 1]
        d_per = l2 * [mask[:,:,num] == 1]
        S0 = xall_unvec[:,:,num,4]*[mask[:,:,num] == 1]
        
        
        #### CALCULATE FA and MD ####
        axial_diffusivity= d_par[0,:,:]
        radial_diffusivity = d_per[0,:,:]
        
        # Define the diffusion tensor using the voxel-wise diffusivity data
        diffusion_tensor = np.zeros((axial_diffusivity.shape[0], axial_diffusivity.shape[1], 3))
        diffusion_tensor[..., 0] = radial_diffusivity
        diffusion_tensor[..., 1] = radial_diffusivity
        diffusion_tensor[..., 2] = axial_diffusivity
        
        
        FA = fractional_anisotropy(diffusion_tensor)
        MD =  (d_par + d_per + d_per) / 3
        
        ## predicted angles
        # Define the size of your image
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
        rgb_map_pred2[:, :, 0] = rgb_map_pred[:, :, 0] * FA   # Red channel
        rgb_map_pred2[:, :, 1] = rgb_map_pred[:, :, 1] * FA   # Green channel
        rgb_map_pred2[:, :, 2] = rgb_map_pred[:, :, 2] * FA   # Blue channel
        
        plt.imshow(np.rot90(rgb_map_pred), cmap='jet')
        plt.title('rgb_map', fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar() 
        plt.show()
        
        plt.imshow(np.rot90(rgb_map_pred2), cmap='jet')
        plt.title('rgb_map * FA map', fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar()  
        plt.show()
        
        plt.figure(figsize=(14, 9), facecolor='black')
        plt.suptitle(f'Results from fitting with LS WITH GNL correction. Dataset: 10{image_idx}', fontsize=25, color='white')
        
        plt.subplot(2, 3, 1)
        plt.imshow(np.rot90(S0[0,:,:]), cmap='gray')
        plt.title('S0', color='white',fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar()                          
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
        plt.setp(cbytick_obj, color='white')
        
        plt.subplot(2, 3, 2)
        plt.imshow(np.rot90(d_par[0,:,:]), cmap='gray')
        plt.title('AD/d_par', color='white',fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar()                          
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
        plt.setp(cbytick_obj, color='white')
        
        plt.subplot(2, 3, 3)
        plt.imshow(np.rot90(d_per[0,:,:]), cmap='gray')
        plt.title('RD/d_per', color='white',fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar()                          
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
        plt.setp(cbytick_obj, color='white')
        
        plt.subplot(2, 3, 4)
        plt.imshow(np.rot90(FA), cmap='gray')
        plt.title('FA map', color='white',fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar()                          
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
        plt.setp(cbytick_obj, color='white')
        
        plt.subplot(2, 3, 5)
        plt.imshow(np.rot90(MD[0,:,:]), cmap='gray' )
        plt.title('MD', color='white',fontsize=15)
        plt.axis('off')
        color_bar = plt.colorbar()                          
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
        plt.setp(cbytick_obj, color='white')
        
        plt.subplot(2, 3, 6)
        plt.imshow(np.rot90(rgb_map_pred2), cmap='jet')
        plt.title('rgb_map * FA map', color='white',fontsize=20)
        plt.axis('off')
        color_bar = plt.colorbar()                          
        cbytick_obj = plt.getp(color_bar.ax.axes, 'yticklabels')            
        plt.setp(cbytick_obj, color='white')
        
        plt.show()
    
    
    ## Save results.
    
    print('... Saving results')
    print('')
    output_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_10' + image_idx + '_dwi_GNL_WITH_correction'
    
    ## save predicted params as a nii flie
    file_path = output_file_name + '_params.nii'
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
        file.write('INFORMATION LS FITTING FOR DATASET WITH GNL INCLUDED')
        file.write('image to fit:' + str(file_path_nii) + '\n')
        file.write('Image shape: ' + str(dwi.shape)+ '\n')
        file.write('B-matrix shape: ' + str(b_matrix.shape)+ '\n')
        file.write('g (bvecs) shape: ' + str(g.shape)+ '\n')
        file.write('grad dev shape: ' + str(grad_dev_.shape)+ '\n' )
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








