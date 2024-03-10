# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:27:46 2023

@author: pcastror

This code fits simulations to the Zeppelin model using Least Squares Optimization. 
Multiple initializations.

THis code saves:
    - estimated parameters from LS fitting as .nii file
    - residuals for each iteration performed as .nii file
    - INFO .txt file with some detaisn of the fitting procedure
    
"""

### Load libraries
import numpy as np
from scipy.optimize import least_squares
import time
import nibabel as nib
import pickle as pk
import torch
from utilities import Zeppelin


print('') 
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('Device used. using GPU (cuda) will significantly speed up the training:', device) # Device used


print('\n\n********** FITTING SIMULATIONS WITH LEAST SQUARES, NO GNL CORRETION **********\n\n')


# Read the image
file_path_nii = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\Simulations\exp1_MRImodelZeppelin_ntrain8000_nval2000_ntest1000_SNR70\syn_sigtest.bin'
file_path_bval  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\bvals.txt '
file_path_bvec  = r'C:\Users\pcastror\Desktop\internship\ML_network\HCP\bvecs.txt '

fh = open(file_path_nii,'rb')
dwi = pk.load(fh)
fh.close()

bvals = np.loadtxt(file_path_bval)
bvecs = np.loadtxt(file_path_bvec)
    
bval=bvals/1000

    
# choose only first 108 bvalues because im using a tensor model
bval = bval[:108]
g = bvecs[:108,:]
     
niter = 5
         
print('\n*************************************************************************************')
print('                                  LOAD DATA TO FIT                                   ')
print('*************************************************************************************\n')
    
    
# choose lower and upper bounds
lb = [0, 0, 0, 0, 0]
ub = [np.pi, 2*np.pi, 3.2, 1, np.inf]
       
print('Image shape: ', dwi.shape)
print('g (bvecs): ',g.shape)
print('bvals: ', bval.shape)
    
    
print('\n*************************************************************************************')
print('                                FTT SIMULATIONS                             ')
print('*************************************************************************************\n')
    
   
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
print('\nFitting...                              \n')
    
    
# Perform model fitting using least squares optimization.
def objective_function(params):
	signal_pred = Zeppelin(params, g, bval)
	signal_pred = signal_pred.reshape(-1)
	residuals = S_voxel - signal_pred
	return residuals
    
    ##################### Fit all slices ###########################################
        
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
print(f'\nExecution time for dataset with num iter {niter}: ', end_time - start_time, ' seconds')
print('\n... Done!\n\n')
    
    
# Find the indices of the minimum values along the second dimension of resnormall
I = np.argmin(resnormall, axis=1)
    
xall_ = np.zeros_like(estimated_params[:, :, 0])
    
# Iterate over voxels and fill xall_ using the indices I
for vox in range(estimated_params.shape[0]):
        xall_[vox, :] = estimated_params[vox, :,  I[vox]]
       
    
print(xall_.shape)
print(resnormall.shape)


## Save results.
    
print('... Saving results\n')
output_file_name = r'C:\Users\pcastror\Desktop\internship\ML_network\FINAL_CODES2\LS_experiments\parameter_estimations\LS_simulation'
    
    ## save predicted params as a nii flie
file_path = output_file_name + '_params.nii'
nii_img = nib.Nifti1Image(xall_, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_img, file_path)
    
print('nii params saved!\n')
    
    
## save residuals as  as a nii flie
file_path = output_file_name + '_res.nii'
nii_res = nib.Nifti1Image(resnormall, affine=np.eye(4))  # Set affine matrix to identity
nib.save(nii_res, file_path)
    
print('nii residuals saved!\n')
    
# File path
ile_path = output_file_name + '_INFO.txt'
    
with open(file_path, 'w') as file:
        file.write('INFORMATION LS FITTING FOR DATASET WITHOUT GNL INCLUDED')
        file.write('image to fit:' + str(file_path_nii) + '\n')
        file.write('Image shape: ' + str(dwi.shape)+ '\n')
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








