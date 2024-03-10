# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:34:16 2024

@author: pcastror

training with multiple datasets.

Input:
    - training, valdiation and test folders
    - read the folders to create the arrays
    - do the training with the arrays

"""

import os
GPU_index = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

### Load libraries
import argparse, os, sys
from numpy import matlib
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
#from torch import autograd
import pickle as pk
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import network_cpu_multiple_datasets

from my_dataset import MyDataset
import time


if __name__ == "__main__":

	print('') 
	device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
	print('Device used. using GPU (cuda) will significantly speed up the training:', device) #Device used: using GPU will significantly speed up the training.


	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program trains a qMRI-net for quantitative MRI parameter estimation. A qMRI-Nnet enables voxel-by-voxel estimation of microstructural properties from sets of MRI images aacquired by varying the MRI sequence parameters.  Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('dtrain', help='path to a pickle binary file storing the input training data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('dval', help='path to a pickle binary file storing the validation data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit (choose among: "pr_hybriddwi" for prostate hybrid diffusion-relaxometry imaging; "br_sirsmdt" for brain saturation recovery diffusion tensor on spherical mean signals; "twocompdwite" for a two-compartment diffusion-t2 relaxation model without anisotropy)). Tissue parameters will be: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0; model "twocompdwite", parameters v, Da, t2a, Db, Kb, t2b, s0')
	parser.add_argument('out_base', help='base name of output directory (a string built with the network parameters will be added to the base). The output directory will contain the following output files: ** losstrain.bin, pickle binary storing the training loss as a numpy matrix (shape: epoch x batch); ** lossval.bin, pickle binary storing the validation loss as a numpy matrix (shape: epoch x 1); ** nnet_epoch0.bin, pickle binary storing the qMRI-net at initialisation; ** nnet_epoch0.pth, Pytorch binary storing the qMRI-net at initialisation; ** nnet_epoch<FINAL_EPOCH>.bin, pickle binary storing the qMRI-net at the final epoch; ** nnet_lossvalmin.bin, pickle binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin.pth, Pytorch binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_sigval.bin, prediction of the validation signals (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_tissueval.bin, prediction of tissue parameters from validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_neuronval.bin, output neuron activations for validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss; ** lossval_min.txt, miniimum validation loss;  ** nnet_lossvalmin_sigtest.bin, prediction of the test signals  (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information), if those signals are provided; ** nnet_lossvalmin_tissuetest.bin, prediction of tissue parameters from test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided; ** nnet_lossvalmin_neurontest.bin, output neuron activations for test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided')
    
	parser.add_argument('--nn', metavar='<list>', help='array storing the number of hidden neurons, separated by hyphens (example: 30-15-8). The first number (input neurons) must equal the number of measurements in the protocol (Nmeas); the last number (output neurons) must equal the number of parameters in the model (Npar, 9 for model "pr_hybriddwi", 4 for model "br_sirsmdt", 7 for model "twocompdwite"). Default: Nmeas-(Npar + (Nmeas minus Npar))/2-Npar, where Nmeas is the number of MRI measurements and Npar is the number of tissue parameters for the signal model to fit.')
	parser.add_argument('--pdrop', metavar='<value>', default='0.0', help='dropout probability in each layer of the neural network. Default: 0.0')
	parser.add_argument('--noepoch', metavar='<value>', default='500', help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', help='number of voxels in each training mini-batch. Default: 1/80 of the total number of training voxels (minimum: 2 voxels)')
	parser.add_argument('--seed', metavar='<value>', default='19102018', help='integer used as a seed for Numpy and PyTorch random number generators. Default: 19102018')
	parser.add_argument('--nwork', metavar='<value>', default='0', help='number of workers for data loader. Default: 0')
	parser.add_argument('--perVAL', metavar='<value>', default='0.15', help='number of workers for data loader. Default: 0')

	parser.add_argument('--dtest', metavar='<file>', help='path to an option input pickle binary file storing the test data as a numpy matrix (rows: voxels; columns: measurements)')
    
	parser.add_argument('--parmin', metavar='<value>', help='list of lower bounds of tissue parameters')
	parser.add_argument('--parmax', metavar='<value>', help='list of upper bounds of tissue parameters.')
	args = parser.parse_args()

	### Get some of the inputs
	pdrop = float(args.pdrop)
	noepoch = int(args.noepoch)
	lrate = float(args.lrate)
	seed = int(args.seed)
	nwork = int(args.nwork)
	mrimodel = args.mri_model

	### Print some information
	print('')
	print('')
	print('********************************************************************')
	print('                 TRAIN A MRI-NET (mrisig CLASS)                   ')
	print('********************************************************************')
	print('')
	print('** Input training data: {}'.format(args.dtrain))
	print('** Input validation data: {}'.format(args.dval))
	if args.dtest is not None:
		print('** Input test data: {}'.format(args.dtest))

    # Paths
	train_path = args.dtrain
	val_path = args.dval
	test_path = args.dtest
    
    
	### TEST DATA
	if args.dtest is not None:
		print('\n\nLOADING TEST DATA ... ')
        
		file_list = os.listdir(test_path)
		test_dictionary = {}

		dwi_files = [file for file in file_list if file.endswith('_dwi.bin')]
		grad_dev_files = [file for file in file_list if file.endswith('_grad_dev.bin')]
		bvals_files = [file for file in file_list if file.endswith('_bvals.bin')]
		bvecs_files = [file for file in file_list if file.endswith('_bvecs.bin')]
		msk_files = [file for file in file_list if file.endswith('_msk.bin')]

		with open(os.path.join(test_path, dwi_files[0]), 'rb') as dwi:    # take this as the reference dataset
			dwi_test = pk.load(dwi) 
            
		with open(os.path.join(test_path, msk_files[0]), 'rb') as msks:
			msk_test = pk.load(msks)
    
		num_slices = dwi_test.shape[2]
		x_shape = dwi_test.shape[0]
		y_shape = dwi_test.shape[1]
		N_meas =  dwi_test.shape[3]

		dwi_A_b0 = dwi_test[:,:,:,0]                                     # choose pixel values from b0 images
		dwi_A_flat = dwi_test[msk_test == 1]                             # array of all pixel values inside the brain
		dwi_A_flat_b0 = dwi_A_b0[msk_test == 1]                          # array of pixel values inside the brain og b0 images
		median_intensity_A  =np.median(dwi_A_flat_b0)                    # extract from b0 images the median
		print(f'\nmedian_intensity reference patient {dwi_files[0]}: ', median_intensity_A)

		max95_patient_A =  np.percentile(dwi_A_flat, 95)                 # 95th per max intensity of patient A (reference patient)
		max_val_test = max95_patient_A

		dwi_test_ = []
		grad_dev_test_ = []
		bvals_test_ = []
		bvecs_test_ = []

    
		# Iterate over the files and load them in one single array
		for i, file in enumerate(dwi_files):
    
			print('')
			print('Dwi file: ', file)

			file_path_dwi = os.path.join(test_path, file)
			file_path_grad = os.path.join(test_path,  grad_dev_files[i])
			file_path_bvals = os.path.join(test_path,  bvals_files[i])
			file_path_bvecs = os.path.join(test_path,  bvecs_files[i])
			file_path_msk = os.path.join(test_path,  msk_files[i])

			with open(file_path_dwi, 'rb') as dwi:
				dwi_test = pk.load(dwi)
        
			with open(file_path_grad, 'rb') as grad_dev:
				grad_dev_test = pk.load(grad_dev)

			with open(file_path_bvals, 'rb') as bvals:
				bvals_test = pk.load(bvals)

			with open(file_path_bvecs, 'rb') as bvecs:
				bvecs_test = pk.load(bvecs) 

			with open(file_path_msk, 'rb') as msks:
				msk_test = pk.load(msks)

			dwi_B_b0 = dwi_test[:,:,:,0]
			dwi_B_flat = dwi_test[msk_test == 1]
			dwi_B_flat_b0 = dwi_B_b0[msk_test == 1]
			median_intensity_B  =np.median(dwi_B_flat_b0)

			scaling_factor =median_intensity_A/ median_intensity_B           # scaling factor between both datasets
			print('\n\tscaling_factor. Median A/B: ', scaling_factor) 


			dwi_B_flat_scaled = dwi_B_flat * scaling_factor                  # scale patient B wrt patient 1 with scaling factor
			normalized_patient_B_sc = dwi_B_flat_scaled/max95_patient_A
        

			bvals_repeated = np.tile(bvals_test, (x_shape,y_shape, 1, 1))
			bvecs_repeated = np.tile(bvecs_test, (x_shape,y_shape, 1, 1, 1))

			grad_dev_test_samples = grad_dev_test[msk_test == 1]
			print('\n\tgrad_dev_test_samples: ', grad_dev_test_samples.shape)

			bvals_test_samples = bvals_repeated[msk_test == 1]
			print('\tbvals_test_samples: ', bvals_test_samples.shape)

			bvecs_test_samples = bvecs_repeated[msk_test == 1]
			print('\tbvecs_test_samples: ', bvecs_test_samples.shape)
            
			print('\tnormalized dwi: ', normalized_patient_B_sc.shape)

			dwi_test_.append(normalized_patient_B_sc)
			grad_dev_test_.append(grad_dev_test_samples)
			bvals_test_.append(bvals_test_samples)
			bvecs_test_.append(bvecs_test_samples)

            
		datatest = np.concatenate(dwi_test_,0)
		grad_dev_test_samples = np.concatenate(grad_dev_test_, 0)
		bvals_test_samples = np.concatenate(bvals_test_, 0)
		bvecs_test_samples = np.concatenate(bvecs_test_, 0)

    
		bvals_corrected_test = bvals_test_samples
		bvecs_corrected_test = bvecs_test_samples
    
		print('\nShape test arrays:\n',datatest.shape)
		print(grad_dev_test_samples.shape)
		print(bvals_corrected_test.shape)
		print(bvecs_corrected_test.shape)

    
    ######## TRAIN DATA #######
	print('\n\nLOADING TRAINING DATA ... ',)

	file_list = os.listdir(train_path)
	train_dictionary = {}

	# Filter the list to keep only the NIfTI files
	dwi_files = [file for file in file_list if file.endswith('_dwi.bin')]
	grad_dev_files = [file for file in file_list if file.endswith('_grad_dev.bin')]
	bvals_files = [file for file in file_list if file.endswith('_bvals.bin')]
	bvecs_files = [file for file in file_list if file.endswith('_bvecs.bin')]
	msk_files = [file for file in file_list if file.endswith('_msk.bin')]

	with open(os.path.join(train_path, dwi_files[0]), 'rb') as dwi:
		dwi_train = pk.load(dwi) 
    
	num_slices = dwi_train.shape[2]
	x_shape = dwi_train.shape[0]
	y_shape = dwi_train.shape[1]
	N_meas =  dwi_train.shape[3]

	dwi_train_ = []
	grad_dev_train_ = []
	bvals_train_ = []
	bvecs_train_ =[]

	for i, file in enumerate(dwi_files):
    
		print('')
		print('Dwi file: ', file)

		file_path_dwi = os.path.join(train_path, file)
		file_path_grad = os.path.join(train_path,  grad_dev_files[i])
		file_path_bvals = os.path.join(train_path,  bvals_files[i])
		file_path_bvecs = os.path.join(train_path,  bvecs_files[i])
		file_path_msk = os.path.join(train_path,  msk_files[i])

		with open(file_path_dwi, 'rb') as dwi:
			dwi_train = pk.load(dwi)
        
		with open(file_path_grad, 'rb') as grad_dev:
			grad_dev_train = pk.load(grad_dev)

		with open(file_path_bvals, 'rb') as bvals:
			bvals_train = pk.load(bvals)

		with open(file_path_bvecs, 'rb') as bvecs:
			bvecs_train = pk.load(bvecs) 

		with open(file_path_msk, 'rb') as msks:
			msk_train = pk.load(msks) 

		dwi_B_b0 = dwi_train[:,:,:,0]
		dwi_B_flat = dwi_train[msk_train == 1]
		dwi_B_flat_b0 = dwi_B_b0[msk_train == 1]
		median_intensity_B  =np.median(dwi_B_flat_b0)
        
		scaling_factor =median_intensity_A/ median_intensity_B           # scaling factor between both datasets
		print('\n\tscaling_factor. Median A/B: ', scaling_factor) 

		dwi_B_flat_scaled = dwi_B_flat * scaling_factor                  # scale patient B wrt patient 1 with scaling factor
		normalized_patient_B_sc = dwi_B_flat_scaled/max95_patient_A
        
            
		bvals_repeated = np.tile(bvals_train, (x_shape, y_shape, 1, 1))
		bvecs_repeated = np.tile(bvecs_train, (x_shape, y_shape, 1, 1, 1))
        

		grad_dev_train_samples = grad_dev_train[msk_train == 1]
		print('\n\tgrad_dev_train_samples: ', grad_dev_train_samples.shape)

		bvals_train_samples = bvals_repeated[msk_train == 1]
		print('\tbvals_train_samples: ', bvals_train_samples.shape)
    
		bvecs_train_samples = bvecs_repeated[msk_train == 1]
		print('\tbvecs_train_samples: ', bvecs_train_samples.shape)
        
		print('\tnormalized dwi: ', normalized_patient_B_sc.shape)

		dwi_train_.append(normalized_patient_B_sc)
		grad_dev_train_.append(grad_dev_train_samples)
		bvals_train_.append(bvals_train_samples)
		bvecs_train_.append(bvecs_train_samples)

	datatrain = np.concatenate(dwi_train_,0)
	grad_dev_train_samples =  np.concatenate(grad_dev_train_,0)
	bvals_train_samples=  np.concatenate(bvals_train_,0)
	bvecs_train_samples =  np.concatenate(bvecs_train_,0)
        
             
	bvals_corrected_train = bvals_train_samples
	bvecs_corrected_train = bvecs_train_samples
    
    
	print('\nShape train arrays:\n', datatrain.shape)
	print(grad_dev_train_samples.shape)
	print(bvals_corrected_train.shape)
	print(bvecs_corrected_train.shape)


    ######## VAL DATA #######

	print('\n\nLOADING VALIDATION DATA ... ')
	file_list = os.listdir(val_path)
	val_dictionary = {}

	dwi_files = [file for file in file_list if file.endswith('_dwi.bin')]
	grad_dev_files = [file for file in file_list if file.endswith('_grad_dev.bin')]
	bvals_files = [file for file in file_list if file.endswith('_bvals.bin')]
	bvecs_files = [file for file in file_list if file.endswith('_bvecs.bin')]
	msk_files = [file for file in file_list if file.endswith('_msk.bin')]

	with open(os.path.join(val_path, dwi_files[0]), 'rb') as dwi:
		dwi_val = pk.load(dwi) 
    
	num_slices = dwi_val.shape[2]
	x_shape = dwi_val.shape[0]
	y_shape = dwi_val.shape[1]
	N_meas =  dwi_val.shape[3]

	dwi_val_ = []
	grad_dev_val_ = []
	bvals_val_ = []
	bvecs_val_ = []
	msk_val_ = []

    
	for i, file in enumerate(dwi_files):
    
		print('\nDwi file: ', file)

		file_path_dwi = os.path.join(val_path, file)
		file_path_grad = os.path.join(val_path,  grad_dev_files[i])
		file_path_bvals = os.path.join(val_path,  bvals_files[i])
		file_path_bvecs = os.path.join(val_path,  bvecs_files[i])
		file_path_msk = os.path.join(val_path,  msk_files[i])

		with open(file_path_dwi, 'rb') as dwi:
			dwi_val = pk.load(dwi)
        
		with open(file_path_grad, 'rb') as grad_dev:
			grad_dev_val = pk.load(grad_dev)

		with open(file_path_bvals, 'rb') as bvals:
			bvals_val = pk.load(bvals)

		with open(file_path_bvecs, 'rb') as bvecs:
			bvecs_val = pk.load(bvecs) 

		with open(file_path_msk, 'rb') as msks:
			msk_val = pk.load(msks) 

		dwi_B_b0 = dwi_val[:,:,:,0]
		dwi_B_flat = dwi_val[msk_val == 1]
		dwi_B_flat_b0 = dwi_B_b0[msk_val == 1]
		median_intensity_B  =np.median(dwi_B_flat_b0)
        
		scaling_factor =median_intensity_A/ median_intensity_B           # scaling factor between both datasets
		print('\n\tscaling_factor. Median A/B: ', scaling_factor) 

		dwi_B_flat_scaled = dwi_B_flat * scaling_factor                  # scale patient B wrt patient 1 with scaling factor
		normalized_patient_B_sc = dwi_B_flat_scaled/max95_patient_A
           
		bvals_repeated = np.tile(bvals_val, (x_shape, y_shape, 1, 1))
		bvecs_repeated = np.tile(bvecs_val, (x_shape, y_shape, 1, 1, 1))
        
		grad_dev_val_samples = grad_dev_val[msk_val == 1]
		print('\n\tgrad_dev_val_samples: ', grad_dev_val_samples.shape)

		bvals_val_samples = bvals_repeated[msk_val == 1]
		print('\tbvals_val_samples: ', bvals_val_samples.shape)

		bvecs_val_samples = bvecs_repeated[msk_val == 1]
		print('\tbvecs_val_samples: ', bvecs_val_samples.shape)

		print('\tnormalized dwi: ', normalized_patient_B_sc.shape)

		dwi_val_.append(normalized_patient_B_sc)
		grad_dev_val_.append(grad_dev_val_samples)
		bvals_val_.append(bvals_val_samples)
		bvecs_val_.append(bvecs_val_samples)

	dataval = np.concatenate(dwi_val_,0)
	grad_dev_val_samples =  np.concatenate(grad_dev_val_,0)
	bvals_val_samples=  np.concatenate(bvals_val_,0)
	bvecs_val_samples =  np.concatenate(bvecs_val_,0)
        
             
	bvals_corrected_val = bvals_val_samples
	bvecs_corrected_val = bvecs_val_samples
    
    
	print('\nShape validation arrays:\n', dataval.shape)
	print(grad_dev_val_samples.shape)
	print(bvals_corrected_val.shape)
	print(bvecs_corrected_val.shape)
        
        
	### Get number of mini-batches
	if args.mbatch is None:
		mbatch = int(float(datatrain.shape[0]) / 80.0)       # Default: 1/80 of the total number of training voxels
		print('\datatrain.shape[0]: ', datatrain.shape[0])
		print('mbatch: ', mbatch)
	else:
		mbatch = int(args.mbatch)
		if (mbatch>datatrain.shape[0]):
			mbatch = datatrain.shape[0]
		if(mbatch<2):
			mbatch = int(2)


	nmeas_train = datatrain.shape[1]
	print('Number of measurements per signal: ', nmeas_train)
	Nmeas = nmeas_train
    
	### Check that MRI model exists
	if ( (mrimodel!='Zeppelin') ):
		raise RuntimeError('the chosen MRI model is not implemented. Sorry!')
	if (mrimodel=='Zeppelin'):
		s0idx = 4


	### Get specifics for hidden layers
	if args.nn is None:

		if (mrimodel=='Zeppelin'): 
			npars = 5                        # number of parameters different in each model
		else:
			raise RuntimeError('the chosen MRI model is not implemented. Sorry!')
            
        # number of hidden neurons
		nhidden = np.array([int(nmeas_train) , int(float(npars)+0.5*( float(nmeas_train) - float(npars))) , int(npars)])
        # this string is used later for the name of the folder where the results will be saved
		nhidden_str = '{}-{}-{}'.format( int(nmeas_train) , int(float(npars)+0.5*( float(nmeas_train) - float(npars))) , int(npars)  )

	else:
		nhidden = (args.nn).split('-')
		nhidden = np.array( list(map( int,nhidden )) )  # number of hidden neurons
		nhidden_str = args.nn
		
	### Get optional user-defined bounds for tissue parameters
	if (args.parmin is not None) or (args.parmax is not None):
		
		if (args.parmin is not None) and (args.parmax is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
		
		if (args.parmax is not None) and (args.parmin is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
		# Lower bound
		pminbound = (args.parmin).split(',')
		pminbound = np.array( list(map( float, pminbound )) )
		# Upper bound
		pmaxbound = (args.parmax).split(',')
		pmaxbound = np.array( list(map( float, pmaxbound )) )


	out_base_dir = '{}_nhidden{}_pdrop{}_noepoch{}_lr{}_mbatch{}_seed{}'.format(args.out_base,nhidden_str,pdrop,noepoch,lrate,mbatch,seed)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir)

	print(f'\n** Output directory: {out_base_dir} \n\n')
	print('PARAMETERS\n')
	print(f'** Hidden neurons: {nhidden}')
	print(f'** Dropout probability: {pdrop}')
	print(f'** Number of epochs: {noepoch}')
	print(f'** Learning rate: {lrate}')
	print(f'** Number of voxels in a mini-batch: {mbatch}')
	print(f'** Seed: {seed}')
	print(f'** Number of workers for data loader: {nwork}\n\n')


	### Set random seeds
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	print('datatrain shape: ', datatrain.shape)
	print('dataval shape: ', dataval.shape)
	print('datatest shape: ', datatest.shape)

    # create an instance of the DATASET class containing the trianing data: pixel vlaues of the dwi image, bvals and bvecs
	datatrain_Class = MyDataset(datatrain, bvals_corrected_train, bvecs_corrected_train)
	loadertrain = DataLoader(datatrain_Class, batch_size=mbatch, shuffle=True, num_workers=nwork)

	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1

	print('\nNumber of mini-batches created: ', nobatch, '\n')
    
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan
    
	nnet = network_cpu_multiple_datasets.mrisig(nhidden, pdrop, mrimodel).to(device)    # Instantiate neural network

	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)       

    # Change tissue parameter ranges
	print('\n** Tissue parameter names: {}'.format(nnet.param_name))
	print('** Tissue parameter lower bounds: {}'.format(nnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(nnet.param_max), '\n\n')	
    
	nnetloss = nn.MSELoss()                                                      # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                      # Network trained with ADAM optimiser
    
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') ) # Save network at epoch 0 (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()


	###################### Run training ###################################

	# Loop over epochs
	start_time = time.time()
	loss_val_prev = np.inf
	for epoch in range(noepoch):
		numk = 0
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch), '\n')

		minibatch_id = 0
		for batch in loadertrain:
            
			dwi_batch = batch['dwi'].to(device)
			bvals_batch = batch['bvals'].to(device)
			bvecs_batch = batch['bvecs'].to(device)
                
			output = nnet(Tensor(dwi_batch), Tensor(bvals_batch), Tensor(bvecs_batch))      # Pass MRI measurements through net and get estimates of MRI signals  
			lossmeas_train = nnetloss(output, dwi_batch)  # Training loss
            
			# Back propagation
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data)
			minibatch_id = minibatch_id + 1
  

		# Run validation
		nnet.eval()   # Set network to evaluation mode (deactivates dropout)
		dataval_nnet = nnet( Tensor(dataval) , Tensor(bvals_corrected_val) , Tensor(bvecs_corrected_val) )     # Output of full network (predicted MRI signals)   
		lossmeas_val = nnetloss( dataval_nnet  , Tensor(dataval)  ) # Validation loss 
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data)

		if(Tensor.numpy(lossmeas_val.data)<=loss_val_prev):

			print('             ... validation loss has decreased. Saving net...')
            
			# Save network
			torch.save( nnet.state_dict(), os.path.join(out_base_dir,'lossvalmin_net.pth') )
			nnet_file = open(os.path.join(out_base_dir,'lossvalmin_net.bin'),'wb')
			pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
			nnet_file.close()
            
			nnet_text = open(os.path.join(out_base_dir,'lossvalmin.info'),'w')
			nnet_text.write('Epoch {} (indices starting from 0)'.format(epoch));
			nnet_text.close();
            
			loss_val_prev = Tensor.numpy(lossmeas_val.data)


			if args.dtest is not None:

				datatest_nnet = nnet( Tensor(datatest) , Tensor(bvals_corrected_test) , Tensor(bvecs_corrected_test)  )     # Output of full network (predicted MRI signals)
				datatest_nnet = datatest_nnet.detach().numpy()
                
				max_val_test_out = np.percentile(datatest_nnet.flatten(), 95)
				tissuetest_nnet = nnet.getparams( Tensor(datatest) )  # Estimated tissue parameters
                      
				# Save predicted test signals
				datatest_nnet = (max_val_test/max_val_test_out)*datatest_nnet # Rescale signals
				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)      
				datatest_nnet_file.close()
                
				# Save predicted test parameters 
				tissuetest_nnet = tissuetest_nnet.detach().numpy()
				tissuetest_nnet[:,s0idx] = (max_val_test/max_val_test_out)*tissuetest_nnet[:,s0idx] # Rescale s0 (any column works)
				tissuetest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissuetest.bin'),'wb')
				pk.dump(tissuetest_nnet,tissuetest_nnet_file,pk.HIGHEST_PROTOCOL)        
				tissuetest_nnet_file.close()
        

		# Set network back to training mode
		nnet.train()

		# Print some information
		print('\n             TRAINING INFO:')
		print('             Training loss: {:.12f}; validation loss: {:.12f}'.format(Tensor.numpy(lossmeas_train.data), Tensor.numpy(lossmeas_val.data)) , '\n')

	end_time = time.time()

	# Save the final network
	nnet.eval()
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch{}_net.pth'.format(noepoch)) )
	nnet_file = open(os.path.join(out_base_dir,'epoch{}_net.bin'.format(noepoch)),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()

	# Save the training and validation loss
	losstrain_file = open(os.path.join(out_base_dir,'losstrain.bin'),'wb')
	pk.dump(losstrain,losstrain_file,pk.HIGHEST_PROTOCOL)      
	losstrain_file.close()

	lossval_file = open(os.path.join(out_base_dir,'lossval.bin'),'wb')
	pk.dump(lossval,lossval_file,pk.HIGHEST_PROTOCOL)      
	lossval_file.close()
	np.savetxt(os.path.join(out_base_dir,'lossval_min.txt'), [np.nanmin(lossval)], fmt='%.12f', delimiter=' ')


	try:
        # File path
		file_path = os.path.join(out_base_dir,'_INFO.txt')

		with open(file_path, 'w') as file:
				file.write('PARAMETERS: ' + '\n')
				file.write('Hidden neurons: ' + str(nhidden) + '\n')
				file.write('Dropout probability: ' + str(pdrop) + '\n')
				file.write('Number of epochs: ' + str(noepoch) + '\n')
				file.write('Learning rate: ' + str(lrate) + '\n')
				file.write('Number of voxels in a mini-batch: ' + str(mbatch)+ '\n')
				file.write('Seed: ' + str(seed) + '\n')
				file.write('Number of workers for data loader: ' + str(nwork) + '\n' + '\n')
                
				file.write('Tissue parameter names: ' + str(nnet.param_name) + '\n')
				file.write('Tissue parameter lower bounds: ' + str(nnet.param_min) + '\n')
				file.write('Tissue parameter upper bounds: ' + str(nnet.param_max) + '\n' + '\n')

				file.write('MRI model: ' + str(mrimodel) + '\n')
                
				file.write('Input measurements path training: ' + str(args.dtrain) + '\n' + '\n')
				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print("\nINFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')
