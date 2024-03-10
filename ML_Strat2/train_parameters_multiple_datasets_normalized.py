# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:28:38 2023

@author: pcastror

Train. Approach 1. train parameters with synsignals.

"""


### Load libraries
import argparse, os, sys
from numpy import matlib
import numpy as np
import torch
import time
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import pickle as pk
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import network_cpu_multiple_datasets
from my_dataset import MyDataset, MyDataset_tensor


if __name__ == "__main__":
    
	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program trains a network for MRI parameter estimation. It enables voxel-by-voxel estimation of microstructural properties from sets of MRI images aacquired by varying the MRI sequence parameters.  Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.College London. All rights reserved.')
	parser.add_argument('dtrain', help='path to a pickle binary file storing the input training MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('dval', help='path to a pickle binary file storing the input validation MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit Zeppelin')
	parser.add_argument('out_base', help='base name of output directory (a string built with the network parameters will be added to the base). The output directory will contain the following output files: ** losstrain.bin, pickle binary storing the training loss as a numpy matrix (shape: epoch x batch); ** lossval.bin, pickle binary storing the validation loss as a numpy matrix (shape: epoch x 1); ** nnet_epoch0.bin, pickle binary storing the qMRI-net at initialisation; ** nnet_epoch0.pth, Pytorch binary storing the qMRI-net at initialisation; ** nnet_epoch<FINAL_EPOCH>.bin, pickle binary storing the qMRI-net at the final epoch; ** nnet_lossvalmin.bin, pickle binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); * nnet_lossvalmin.pth, Pytorch binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_sigval.bin, prediction of the validation signals (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin_tissueval.bin, prediction of tissue parameters from validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); ** nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss; ** lossval_min.txt, miniimum validation loss; ** nnet_lossvalmin_sigtest.bin, prediction of the test signals  (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information), if those signals are provided; ** nnet_lossvalmin_tissuetest.bin, prediction of tissue parameters from test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided')
	parser.add_argument('--nn', metavar='<list>', help='array storing the number of hidden neurons, separated by hyphens (example: 30-15-8). The first number (input neurons) must equal the number of measurements in the protocol (Nmeas); the last number (output neurons) must equal the number of parameters in the model (Npar, 9 for model "pr_hybriddwi", 4 for model "br_sirsmdt", 7 for model "twocompdwite"). Default: Nmeas-(Npar + (Nmeas minus Npar))/2-Npar, where Nmeas is the number of MRI measurements and Npar is the number of tissue parameters for the signal model to fit')
	parser.add_argument('--pdrop', metavar='<value>', default='0.0', help='dropout probability in each layer of the neural network. Default: 0.0')
	parser.add_argument('--noepoch', metavar='<value>', default='100', help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', help='number of voxels in each training mini-batch. Default: 1/80 of the total number of training voxels (minimum: 2 voxels)')
	parser.add_argument('--seed', metavar='<value>', default='19102018', help='integer used as a seed for Numpy and PyTorch random number generators. Default: 19102018')
	parser.add_argument('--nwork', metavar='<value>', default='0', help='number of workers for data loader. Default: 0')
	parser.add_argument('--dtest', metavar='<file>', help='path to an option input pickle binary file storing test MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('--parmin', metavar='<value>', help='list of lower bounds of tissue parameters.')
	parser.add_argument('--parmax', metavar='<value>', help='list of upper bounds of tissue parameters. ')
	args = parser.parse_args()

	### Get some of the inputs
	pdrop = float(args.pdrop)      # dropout probability
	noepoch = int(args.noepoch)    # number of epoch
	lrate = float(args.lrate)      # learning rate
	seed = int(args.seed)          # integer used as a seed for Numpy and PyTorch random number generators
	nwork = int(args.nwork)        # number of workers in dataloader
	mrimodel = args.mri_model      # mri model

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
		params_files = [file for file in file_list if file.endswith('_params.bin')]
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
		params_test_ = []
		grad_dev_test_ = []
		bvals_test_ = []
		bvecs_test_ = []

    
		# Iterate over the files and load them in one single array
		for i, file in enumerate(dwi_files):
    
			print('')
			print('Dwi file: ', file)

			file_path_dwi = os.path.join(test_path, file)
			file_path_params = os.path.join(test_path,  params_files[i])
			file_path_grad = os.path.join(test_path,  grad_dev_files[i])
			file_path_bvals = os.path.join(test_path,  bvals_files[i])
			file_path_bvecs = os.path.join(test_path,  bvecs_files[i])
			file_path_msk = os.path.join(test_path,  msk_files[i])

			with open(file_path_dwi, 'rb') as dwi:
				dwi_test = pk.load(dwi)
				
			with open(file_path_params, 'rb') as params:
				params_test = pk.load(params)
        
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
			
			params_test_samples = params_test[msk_test == 1]
			print('\n\tparams_test_samples: ', params_test_samples.shape)

			bvals_test_samples = bvals_repeated[msk_test == 1]
			print('\tbvals_test_samples: ', bvals_test_samples.shape)

			bvecs_test_samples = bvecs_repeated[msk_test == 1]
			print('\tbvecs_test_samples: ', bvecs_test_samples.shape)
            
			print('\tnormalized dwi: ', normalized_patient_B_sc.shape)

			dwi_test_.append(normalized_patient_B_sc)
			params_test_.append(params_test_samples)
			grad_dev_test_.append(grad_dev_test_samples)
			bvals_test_.append(bvals_test_samples)
			bvecs_test_.append(bvecs_test_samples)

            
		datatest = np.concatenate(dwi_test_,0)
		paramstest = np.concatenate(params_test_,0)
		grad_dev_test_samples = np.concatenate(grad_dev_test_, 0)
		bvals_test_samples = np.concatenate(bvals_test_, 0)
		bvecs_test_samples = np.concatenate(bvecs_test_, 0)

    
		bvals_corrected_test = bvals_test_samples
		bvecs_corrected_test = bvecs_test_samples
    
		print('\nShape test arrays:\n',datatest.shape)
		print(paramstest.shape)
		print(grad_dev_test_samples.shape)
		print(bvals_corrected_test.shape)
		print(bvecs_corrected_test.shape)

    
    ######## TRAIN DATA #######
	print('\n\nLOADING TRAINING DATA ... ',)

	file_list = os.listdir(train_path)
	train_dictionary = {}

	# Filter the list to keep only the NIfTI files
	dwi_files = [file for file in file_list if file.endswith('_dwi.bin')]
	params_files = [file for file in file_list if file.endswith('_params.bin')]
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
	params_train_ = []
	grad_dev_train_ = []
	bvals_train_ = []
	bvecs_train_ =[]

	for i, file in enumerate(dwi_files):
    
		print('')
		print('Dwi file: ', file)

		file_path_dwi = os.path.join(train_path, file)
		file_path_params = os.path.join(train_path,  params_files[i])
		file_path_grad = os.path.join(train_path,  grad_dev_files[i])
		file_path_bvals = os.path.join(train_path,  bvals_files[i])
		file_path_bvecs = os.path.join(train_path,  bvecs_files[i])
		file_path_msk = os.path.join(train_path,  msk_files[i])

		with open(file_path_dwi, 'rb') as dwi:
			dwi_train = pk.load(dwi)
			
		with open(file_path_params, 'rb') as params:
			params_train = pk.load(params)
        
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

		params_train_samples = params_train[msk_train == 1]
		print('\n\tparams_train_samples: ', params_train_samples.shape)
		
		bvals_train_samples = bvals_repeated[msk_train == 1]
		print('\tbvals_train_samples: ', bvals_train_samples.shape)
    
		bvecs_train_samples = bvecs_repeated[msk_train == 1]
		print('\tbvecs_train_samples: ', bvecs_train_samples.shape)
        
		print('\tnormalized dwi: ', normalized_patient_B_sc.shape)

		dwi_train_.append(normalized_patient_B_sc)
		params_train_.append(params_train_samples)
		grad_dev_train_.append(grad_dev_train_samples)
		bvals_train_.append(bvals_train_samples)
		bvecs_train_.append(bvecs_train_samples)

	datatrain = np.concatenate(dwi_train_,0)
	paramstrain = np.concatenate(params_train_,0)
	grad_dev_train_samples =  np.concatenate(grad_dev_train_,0)
	bvals_train_samples=  np.concatenate(bvals_train_,0)
	bvecs_train_samples =  np.concatenate(bvecs_train_,0)
        
             
	bvals_corrected_train = bvals_train_samples
	bvecs_corrected_train = bvecs_train_samples
    
    
	print('\nShape train arrays:\n', datatrain.shape)
	print(paramstrain.shape)
	print(grad_dev_train_samples.shape)
	print(bvals_corrected_train.shape)
	print(bvecs_corrected_train.shape)


    ######## VAL DATA #######

	print('\n\nLOADING VALIDATION DATA ... ')
	file_list = os.listdir(val_path)
	val_dictionary = {}

	dwi_files = [file for file in file_list if file.endswith('_dwi.bin')]
	params_files = [file for file in file_list if file.endswith('_params.bin')]
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
	params_val_ = []
	grad_dev_val_ = []
	bvals_val_ = []
	bvecs_val_ = []
	msk_val_ = []

    
	for i, file in enumerate(dwi_files):
    
		print('\nDwi file: ', file)

		file_path_dwi = os.path.join(val_path, file)
		file_path_params = os.path.join(val_path,  params_files[i])
		file_path_grad = os.path.join(val_path,  grad_dev_files[i])
		file_path_bvals = os.path.join(val_path,  bvals_files[i])
		file_path_bvecs = os.path.join(val_path,  bvecs_files[i])
		file_path_msk = os.path.join(val_path,  msk_files[i])

		with open(file_path_dwi, 'rb') as dwi:
			dwi_val = pk.load(dwi)
			
		with open(file_path_params, 'rb') as params:
			params_val = pk.load(params)
        
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
		
		params_val_samples = params_val[msk_val == 1]
		print('\n\tparams_val_samples: ', params_val_samples.shape)

		bvals_val_samples = bvals_repeated[msk_val == 1]
		print('\tbvals_val_samples: ', bvals_val_samples.shape)

		bvecs_val_samples = bvecs_repeated[msk_val == 1]
		print('\tbvecs_val_samples: ', bvecs_val_samples.shape)

		print('\tnormalized dwi: ', normalized_patient_B_sc.shape)

		dwi_val_.append(normalized_patient_B_sc)
		params_val_.append(params_val_samples)
		grad_dev_val_.append(grad_dev_val_samples)
		bvals_val_.append(bvals_val_samples)
		bvecs_val_.append(bvecs_val_samples)

	dataval = np.concatenate(dwi_val_,0)
	paramsval = np.concatenate(params_val_,0)
	grad_dev_val_samples =  np.concatenate(grad_dev_val_,0)
	bvals_val_samples=  np.concatenate(bvals_val_,0)
	bvecs_val_samples =  np.concatenate(bvecs_val_,0)
        
             
	bvals_corrected_val = bvals_val_samples
	bvecs_corrected_val = bvecs_val_samples
    
    
	print('\nShape validation arrays:\n', dataval.shape)
	print(paramsval.shape)
	print(grad_dev_val_samples.shape)
	print(bvals_corrected_val.shape)
	print(bvecs_corrected_val.shape)
	
	prmtrain = paramstrain
	npar_train = prmtrain.shape[1]
	
	if prmtrain.shape[0]!=datatrain.shape[0]:
		raise RuntimeError('the number of voxels in the training parameters differs from the training MRI signals!')		

	### Load validation tissue parameters
	prmval = paramsval
	if prmval.shape[0]!=dataval.shape[0]:
		raise RuntimeError('the number of voxels in the validation parameters differs from the validation MRI signals!')
	if prmval.shape[1]!=prmtrain.shape[1]:
		raise RuntimeError('the number of validation parameters differs from the number of training parameters!')		


	### Get number of mini-batches
	if args.mbatch is None:
		mbatch = int(float(datatrain.shape[0]) / 80.0) # Default: 1/80 of the total number of training voxels
		print('mbatch: ', mbatch)
	else:
		mbatch = int(args.mbatch)
		if (mbatch>datatrain.shape[0]):
			mbatch = datatrain.shape[0]
		if(mbatch<2):
			mbatch = int(2)

	nmeas_train = datatrain.shape[1]

	### Check that MRI model exists
	if ( (mrimodel!='Zeppelin') ):
		raise RuntimeError('the chosen MRI model is not implemented. Sorry!')
	if (mrimodel=='Zeppelin'):
		s0idx = 4     # index of S0 signal. It is in the last position!!!


	### Get specifics for hidden layers
	if args.nn is None:
		if (mrimodel=='Zeppelin'): 
			npars = 5                        # number of parameters in the model
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

	### Get optional user-defined bounds for tissue parameters. if new bounds provided, change them.
	if (args.parmin is not None) or (args.parmax is not None):
		
		if (args.parmin is not None) and (args.parmax is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
		
		if (args.parmax is not None) and (args.parmin is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
					
		# Lower bounds
		pminbound = (args.parmin).split(',')
		pminbound = np.array( list(map( float, pminbound )) )
		
		# Upper bounds
		pmaxbound = (args.parmax).split(',')
		pmaxbound = np.array( list(map( float, pmaxbound )) )


	### Create output base name
	out_base_dir = '{}_Strat1_nhidden{}_pdrop{}_noepoch{}_lr{}_mbatch{}_seed{}'.format(args.out_base,nhidden_str,pdrop,noepoch,lrate,mbatch,seed)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir) # create the folder if it does not exist

	print(f'\n** Output directory: {out_base_dir} \n\n')
	print('PARAMETERS\n')
	print(f'** Hidden neurons: {nhidden}')
	print(f'** Dropout probability: {pdrop}')
	print(f'** Number of epochs: {noepoch}')
	print(f'** Learning rate: {lrate}')
	print(f'** Number of voxels in a mini-batch: {mbatch}')
	print(f'** Seed: {seed}')
	print(f'** Number of workers for data loader: {nwork}\n\n')


	### Set random seeds.
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	#ensure that the data is in the range [0, 1]. Loose S0 influnce on paramters to give all params the same importance in MSE 
	max_val_train_param = np.max(prmtrain[:, s0idx])
	prmtrain[:,s0idx] = np.float32( prmtrain[:,s0idx] / max_val_train_param )
	max_val_val_param = np.max(prmval[:, s0idx])
	prmval[:,s0idx] = np.float32( prmval[:,s0idx] / max_val_val_param )
	
	print('datatrain shape: ', datatrain.shape)
	print('dataval shape: ', dataval.shape)
	print('datatest shape: ', datatest.shape)
	
    # make their size multiple of 100
	prmtrain = Tensor(prmtrain[prmtrain.shape[0]%100:,:])
	datatrain = Tensor(datatrain[datatrain.shape[0]%100:,:])

	print('datatrain :', datatrain.shape)
	print('prmtrain shape:', prmtrain.shape)
    
	combined_tensor = torch.cat((datatrain, prmtrain),dim=1)
	print('combined_tensor: ', combined_tensor.shape)
    
	### Create mini-batches on training data with data loader.
	loadertrain = DataLoader(combined_tensor, batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses.
	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1
	print('Number of mini-batches created: ', nobatch, '\n')	

	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan

	### Instantiate the network and training objects, and save the intantiated network
	nnet = network_cpu_multiple_datasets.mripar(nhidden, pdrop, mrimodel).to('cpu')   # Instantiate neural network in cpu
    
   # Change tissue parameter ranges if specified
	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)  
        
	print('\n** Tissue parameter names: {}'.format(nnet.param_name))
	print('** Tissue parameter lower bounds: {}'.format(nnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(nnet.param_max), '\n\n')	
    
	nnetloss = nn.MSELoss()                                                         # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                         # Network trained with ADAM optimiser
    
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') )    # Save network at epoch 0 as a state distionary (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()


	slope_norm_tr = np.ones((mbatch , npar_train))
	offset_norm_tr = np.ones((mbatch , npar_train))

   # This loop calculates the normalization factors for each parameter in the model. 
	for pp in range(0,npar_train):
		slope_norm_tr[:,pp] = 1.0 / (nnet.param_max[pp] - nnet.param_min[pp])
		offset_norm_tr[:,pp] = (-1.0*nnet.param_min[pp]) / (nnet.param_max[pp] - nnet.param_min[pp])

   # convert them to Pytorch tensors
	slope_norm_tr = Tensor(np.float32(slope_norm_tr))
	offset_norm_tr = Tensor(np.float32(offset_norm_tr))

	nvox_val = dataval.shape[0]
   # Repeat the same for the validation data
	slope_norm_val = np.ones((nvox_val , npar_train))
	offset_norm_val = np.ones((nvox_val , npar_train))

	for pp in range(0,npar_train):
		slope_norm_val[:,pp] = 1.0 / (nnet.param_max[pp] - nnet.param_min[pp])
		offset_norm_val[:,pp] = (-1.0*nnet.param_min[pp]) / (nnet.param_max[pp] - nnet.param_min[pp])

	slope_norm_val = Tensor(np.float32(slope_norm_val))
	offset_norm_val = Tensor(np.float32(offset_norm_val))
    
    
	###################### Run training ###################################
	# Loop over epochs
	start_time = time.time()

	loss_val_prev = np.inf
	for epoch in range(noepoch):
	    
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch), '\n')

		# Loop over mini-batches for at a fixed epoch
		minibatch_id = 0
		for dwi_batch in loadertrain:
			output = nnet(dwi_batch[:,0:nmeas_train] )  # Pass MRI measurements and estimate tissue parmaters

			try:
                #The training loss is calculated by comparing the estimated tissue parameters (output) with the ground truth tissue parameters
				lossmeas_train = nnetloss(output*slope_norm_tr + offset_norm_tr, dwi_batch[:,nmeas_train:nmeas_train+npar_train]*slope_norm_tr + offset_norm_tr) # Training loss 
			except:
				raise RuntimeError('The number of training voxels must be a multiple of the size of the mini-batch!')

			# Back propagation. 
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			# Store loss for the current mini-batch of training
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data.detach().clone().cpu())

			# Update mini-batch counter
			minibatch_id = minibatch_id + 1
		
		# Run validation. 
		nnet.eval()   # Set network to evaluation mode (deactivates dropout)
		tissueval_nnet = nnet( Tensor(dataval))                  # Output of full network (predicted tissue parameters)
		lossmeas_val = nnetloss( tissueval_nnet*slope_norm_val + offset_norm_val  , Tensor(prmval)*slope_norm_val + offset_norm_val  ) # Validation loss
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data.detach().clone().cpu())

		if(Tensor.numpy(lossmeas_val.data.detach().clone().cpu())<=loss_val_prev):
			print('             ... validation loss has decreased. Saving net...')
			
			# Save network
			torch.save( nnet.state_dict(), os.path.join(out_base_dir,'lossvalmin_net.pth') )
			nnet_file = open(os.path.join(out_base_dir,'lossvalmin_net.bin'),'wb')
			pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
			nnet_file.close()
			
			# Save information on the epoch
			nnet_text = open(os.path.join(out_base_dir,'lossvalmin.info'),'w')
			nnet_text.write('Epoch {} (indices starting from 0)'.format(epoch));
			nnet_text.close();
			
			# Update value of best validation loss so far
			loss_val_prev = Tensor.numpy(lossmeas_val.data)
			
			if args.dtest is not None:

				tissuetest_nnet = nnet( Tensor(datatest) )              # Output of network (estimated tissue parameters)
                
				datatest_nnet = nnet.getsignals( tissuetest_nnet , Tensor(bvals_corrected_test) , Tensor(bvecs_corrected_test)) # Predicted MRI signals
				datatest_nnet = datatest_nnet.detach().cpu().numpy()
				max_val_test_out = np.percentile(datatest_nnet.flatten(), 95)
                
				# Save predicted test tissue parameters 
				tissuetest_nnet = tissuetest_nnet.detach().numpy()

				tissuetest_nnet[:,s0idx] = (max_val_test/max_val_test_out)*tissuetest_nnet[:,s0idx] # Rescale s0
				tissuetest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissuetest.bin'),'wb')
				pk.dump(tissuetest_nnet,tissuetest_nnet_file,pk.HIGHEST_PROTOCOL)      
				tissuetest_nnet_file.close()
                
				# Save predicted test signals
				datatest_nnet = (max_val_test/max_val_test_out)*datatest_nnet # Rescale signal	
				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)         
				datatest_nnet_file.close()


		# Set network back to training mode
		nnet.train()

		# Print some information
		print('\n             TRAINING INFO:')
		print('             Training loss: {:.12f}; validation loss: {:.12f}'.format(
            Tensor.numpy(lossmeas_train.data), 
            Tensor.numpy(lossmeas_val.data)) , '\n')

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
                
				file.write('Input signals path: ' + str(args.dtrain) + '\n' + '\n')
				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')

