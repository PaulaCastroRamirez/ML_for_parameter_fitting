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
	parser.add_argument('sig_train', help='path to a pickle binary file storing the input training MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('sig_val', help='path to a pickle binary file storing the input validation MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
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
	print('                 TRAIN A MRI-NET (mripar CLASS)                   ')
	print('********************************************************************')
	print('')
	print('** Input training MRI signals: {}'.format(args.sig_train))
	print('** Input validation MRI signals: {}'.format(args.sig_val))
	if args.dtest is not None:
		print('** Input test data: {}'.format(args.dtest))

    # Paths
	train_path = args.sig_train
	val_path = args.sig_val
	test_path = args.dtest
    
    ######## TRAIN DATA #######
	print('')
	print('')
	print('LOADING TRAINING DATA ... ',)

    # Get a list of all the files in the folder
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

	dwi_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas))
	grad_dev_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 9))
	params_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 5))
	bvals_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas ))
	bvecs_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas, 3))
	msk_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files)))


	slices2 = 0
    
	# Iterate over the files and load them in one single array
	for i, file in enumerate(dwi_files):
    
		print('')
		print('Num dwi file: ', i)

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
			print(params_train.shape)
        
		with open(file_path_grad, 'rb') as grad_dev:
			grad_dev_train = pk.load(grad_dev)

		with open(file_path_bvals, 'rb') as bvals:
			bvals_train = pk.load(bvals)

		with open(file_path_bvecs, 'rb') as bvecs:
			bvecs_train = pk.load(bvecs) 

		with open(file_path_msk, 'rb') as msks:
			msk_train = pk.load(msks) 
         
		slices1 = slices2
		slices2 = dwi_train.shape[2] + slices1
    
		print('From Slice: ', slices1, ' To slice: ', slices2)
    
		bvals_repeated = np.tile(bvals_train, (x_shape, y_shape, 1, 1))
		bvecs_repeated = np.tile(bvecs_train, (x_shape, y_shape, 1, 1, 1))
		print('bvals_repeated: ', bvals_repeated.shape)
		print('bvecs_repeated: ', bvecs_repeated.shape)

		dwi_train_[:,:,slices1:slices2,:]       = dwi_train
		params_train_[:,:,slices1:slices2,:]    = params_train
		grad_dev_train_[:,:,slices1:slices2,:]  = grad_dev_train
		bvals_train_[:,:,slices1:slices2, :]    = bvals_train
		bvecs_train_[:,:,slices1:slices2, :,:]  = bvecs_train
		msk_train_[:,:,slices1:slices2]         = msk_train


	print('')
	print('dwi_train_: ', dwi_train_.shape)
	print('params_train_: ', params_train_.shape)
	print('grad_dev_train_: ', grad_dev_train_.shape)
	print('bvals_train_: ', bvals_train_.shape)
	print('bvecs_train_: ', bvecs_train_.shape)
	print('msk_train_: ', msk_train_.shape)

	print('')
	dwi_train_samples = dwi_train_[msk_train_ == 1]
	print('dwi_train_samples: ', dwi_train_samples.shape)
	
	params_train_samples = np.nan_to_num(params_train_[msk_train_ == 1])
	print('params_train_samples: ', params_train_samples.shape)

	grad_dev_train_samples = grad_dev_train_[msk_train_ == 1]
	print('grad_dev_train_samples: ', grad_dev_train_samples.shape)

	bvals_train_samples = bvals_train_[msk_train_ == 1]
	print('bvals_train_samples: ', bvals_train_samples.shape)
    
	bvecs_train_samples = bvecs_train_[msk_train_ == 1]
	print('bvecs_train_samples: ', bvecs_train_samples.shape)

	bvals_corrected_train = bvals_train_samples
	bvecs_corrected_train = bvecs_train_samples


    ######## VAL DATA #######
	print('')
	print('')
	print('LOADING VALIDATION DATA ... ')
	print('')

    # Get a list of all the files in the folder
	file_list = os.listdir(val_path)
	val_dictionary = {}

	# Filter the list to keep only the NIfTI files
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

	dwi_val_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas))
	params_val_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 5))
	grad_dev_val_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 9))
	bvals_val_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas ))
	bvecs_val_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas, 3))
	msk_val_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files)))

	print('dwi_val_: ', dwi_val_.shape)
	print('params_val_: ', params_val_.shape)
	print('grad_dev_val_: ', grad_dev_val_.shape)
	print('bvals_val_: ', bvals_val_.shape)
	print('bvecs_val_: ', bvecs_val_.shape)
	print('bvecs_val_: ', msk_val_.shape)

	slices2 = 0
    
	# Iterate over the files and load them in one single array
	for i, file in enumerate(dwi_files):
    
		print('')
		print('Num dwi file: ', i)

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
         
		slices1 = slices2
		slices2 = dwi_val.shape[2] + slices1
    
		print('From Slice: ', slices1, ' To slice: ', slices2)

            
		bvals_repeated = np.tile(bvals_val, (x_shape, y_shape, 1, 1))
		bvecs_repeated = np.tile(bvecs_val, (x_shape, y_shape, 1, 1, 1))
        
		print('bvals_repeated: ', bvals_repeated.shape)
		print('bvecs_repeated: ', bvecs_repeated.shape)

		dwi_val_[:,:,slices1:slices2,:]       = dwi_val
		params_val_[:,:,slices1:slices2,:]    = params_val
		grad_dev_val_[:,:,slices1:slices2,:]  = grad_dev_val
		bvals_val_[:,:,slices1:slices2, :]    = bvals_val
		bvecs_val_[:,:,slices1:slices2, :,:]  = bvecs_val
		msk_val_[:,:,slices1:slices2]         = msk_val

		print('')
		print('dwi_val_: ', dwi_val_.shape)
		print('params_val_: ', params_val_.shape)
		print('grad_dev_val_: ', grad_dev_val_.shape)
		print('bvals_val_: ', bvals_val_.shape)
		print('bvecs_val_: ', bvecs_val_.shape)
		print('msk_val_: ', msk_val_.shape)

	print('')
	dwi_val_samples = dwi_val_[msk_val_ == 1]
	print('dwi_val_samples: ', dwi_val_samples.shape)
	
	params_val_samples =  np.nan_to_num(params_val_[msk_val_ == 1])
	print('params_val_samples: ', params_val_samples.shape)

	grad_dev_val_samples = grad_dev_val_[msk_val_ == 1]
	print('grad_dev_val_samples: ', grad_dev_val_samples.shape)

	bvals_val_samples = bvals_val_[msk_val_ == 1]
	print('bvals_val_samples: ', bvals_val_samples.shape)

	bvecs_val_samples = bvecs_val_[msk_val_ == 1]
	print('bvecs_val_samples: ', bvecs_val_samples.shape)


	bvals_corrected_val = bvals_val_samples
	bvecs_corrected_val = bvecs_val_samples
    
    

	### TEST DATA
	if args.dtest is not None:
    	######## TEST DATA #######
		print('')
		print('')
		print('LOADING TEST DATA ... ')
		
    	# Get a list of all the files in the folder
		file_list = os.listdir(test_path)
		test_dictionary = {}

		# Filter the list to keep only the NIfTI files
		dwi_files = [file for file in file_list if file.endswith('_dwi.bin')]
		params_files = [file for file in file_list if file.endswith('_params.bin')]
		grad_dev_files = [file for file in file_list if file.endswith('_grad_dev.bin')]
		bvals_files = [file for file in file_list if file.endswith('_bvals.bin')]
		bvecs_files = [file for file in file_list if file.endswith('_bvecs.bin')]
		msk_files = [file for file in file_list if file.endswith('_msk.bin')]

		with open(os.path.join(test_path, dwi_files[0]), 'rb') as dwi:
			dwi_test = pk.load(dwi) 
    
		num_slices = dwi_test.shape[2]
		x_shape = dwi_test.shape[0]
		y_shape = dwi_test.shape[1]
		N_meas =  dwi_test.shape[3]

		dwi_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas))
		params_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 5))
		grad_dev_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 9))
		bvals_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas ))
		bvecs_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas, 3))
		msk_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files)))

		print('')
		print('dwi_test_: ', dwi_test_.shape)
		print('params_test_: ', params_test_.shape)
		print('grad_dev_test_: ', grad_dev_test_.shape)
		print('bvals_test_: ', bvals_test_.shape)
		print('bvecs_test_: ', bvecs_test_.shape)
		print('bvecs_test_: ', msk_test_.shape)

		slices2 = 0
    
		# Iterate over the files and load them in one single array
		for i, file in enumerate(dwi_files):
    
			print('')
			print('Num dwi file: ', i)

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
				#msk_test = np.array(pk.load(msks))
				msk_test = pk.load(msks)

			slices1 = slices2
			slices2 = dwi_test.shape[2] + slices1
    
			print('From Slice: ', slices1, ' To slice: ', slices2)
        

			bvals_repeated = np.tile(bvals_test, (x_shape, y_shape, 1, 1))
			bvecs_repeated = np.tile(bvecs_test, (x_shape, y_shape, 1, 1, 1))
			print('bvals_repeated: ', bvals_repeated.shape)
			print('bvecs_repeated: ', bvecs_repeated.shape)

			dwi_test_[:,:,slices1:slices2,:]       = dwi_test
			params_test_[:,:,slices1:slices2,:]  = params_test
			grad_dev_test_[:,:,slices1:slices2,:]  = grad_dev_test
			bvals_test_[:,:,slices1:slices2, :]    = bvals_test
			bvecs_test_[:,:,slices1:slices2, :,:]  = bvecs_test
			msk_test_[:,:,slices1:slices2]         = msk_test

		print('')
		print('dwi_test_: ', dwi_test_.shape)
		print('params_test_: ', params_test_.shape)
		print('grad_dev_test_: ', grad_dev_test_.shape)
		print('bvals_test_: ', bvals_test_.shape)
		print('bvecs_test_: ', bvecs_test_.shape)
		print('msk_test_: ', msk_test_.shape)

		print('')
		dwi_test_samples = dwi_test_[msk_test_ == 1]
		print('dwi_test_samples: ', dwi_test_samples.shape)

		params_test_samples =  np.nan_to_num(params_test_[msk_test_ == 1])
		print('params_test_samples: ', params_test_samples.shape)
		
		grad_dev_test_samples = grad_dev_test_[msk_test_ == 1]
		print('grad_dev_test_samples: ', grad_dev_test_samples.shape)

		bvals_test_samples = bvals_test_[msk_test_ == 1]
		print('bvals_test_samples: ', bvals_test_samples.shape)

		bvecs_test_samples = bvecs_test_[msk_test_ == 1]
		print('bvecs_test_samples: ', bvecs_test_samples.shape)
        
		#datatest = np.concatenate((dwi_test_samples, grad_dev_test_samples), axis=1)
        
		bvals_corrected_test = bvals_test_samples
		bvecs_corrected_test = bvecs_test_samples
		
	
	### Load training tissue parameters
	prmtrain = params_train_samples
	print(prmtrain[789009,3])
	npar_train = prmtrain.shape[1]
	#print(prmtrain.shape[0])
	if prmtrain.shape[0]!=dwi_train_samples.shape[0]:
		raise RuntimeError('the number of voxels in the training parameters differs from the training MRI signals!')		

	### Load validation tissue parameters
	prmval = params_val_samples
	if prmval.shape[0]!=dwi_val_samples.shape[0]:
		raise RuntimeError('the number of voxels in the validation parameters differs from the validation MRI signals!')
	if prmval.shape[1]!=prmtrain.shape[1]:
		raise RuntimeError('the number of validation parameters differs from the number of training parameters!')		

	datatrain = dwi_train_samples
	dataval = dwi_val_samples
	datatest = dwi_test_samples
    
	### Get number of mini-batches
	if args.mbatch is None:
		mbatch = int(float(datatrain.shape[0]) / 80.0) # Default: 1/80 of the total number of training voxels
		print('datatrain.shape[0]: ', datatrain.shape[0])
		print('mbatch: ', mbatch)
	else:
		mbatch = int(args.mbatch)
		if (mbatch>datatrain.shape[0]):
			mbatch = datatrain.shape[0]
		if(mbatch<2):
			mbatch = int(2)

	Nmeas = dwi_train_samples.shape[1]
	nmeas_train = dwi_train_samples.shape[1]

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

	### Print some more information
	print('** Output directory: {}'.format(out_base_dir))
	print('')
	print('')
	print('PARAMETERS')
	print('')
	print('** Hidden neurons: {}'.format(nhidden))
	print('** Dropout probability: {}'.format(pdrop))
	print('** Number of epochs: {}'.format(noepoch))
	print('** Learning rate: {}'.format(lrate))
	print('** Number of voxels in a mini-batch: {}'.format(mbatch))
	print('** Seed: {}'.format(seed))
	print('** Number of workers for data loader: {}'.format(nwork))
	print('')
	print('')


	### Set random seeds. By setting the same seed, you ensure that the random numbers generated in the code will be the same each time you run it
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	### Normalise MRI signals and convert to single precision. Normalize the datatrain array by dividing it by the maximum value along each row. 
   # It performs element-wise division to ensure that the data is in the range [0, 1].
	max_val_train = np.transpose( matlib.repmat(np.max(datatrain,axis=1),nmeas_train,1) )
	for m, i in enumerate(max_val_train):
        	if i.any() == 0:
        		max_val_train[m]= np.ones_like(max_val_train[m])
				
	datatrain = np.float32( datatrain / max_val_train )
    

   # to ensure that the data is in the range [0, 1].
	max_val_train_param = np.max(prmtrain[:, s0idx])
	prmtrain[:,s0idx] = np.float32( prmtrain[:,s0idx] / max_val_train_param )
        
   # normalize the dataval array using the maximum values along each row.
	max_val_val = np.transpose( matlib.repmat(np.max(dataval,axis=1),nmeas_train,1) ) 
	for m, i in enumerate(max_val_val):
        	if i.any() == 0:
        		max_val_val[m]= np.ones_like(max_val_val[m])
				
	dataval = np.float32( dataval / max_val_val )
	dataval = torch.from_numpy(dataval)
    
   #ensure that the data is in the range [0, 1].
	max_val_val_param = np.max(prmval[:, s0idx])
	prmval[:,s0idx] = np.float32( prmval[:,s0idx] / max_val_val_param )
	prmval = torch.from_numpy(prmval)

    # If it is not None, it means there is test data available, and the code normalizes the datatest array using the maximum values along each row.
	if args.dtest is not None:
		max_val_test = np.transpose( matlib.repmat(np.max(datatest,axis=1),nmeas_train,1) )
		for m, i in enumerate(max_val_test):
			if i.any() == 0:
				max_val_test[m]= np.ones_like(max_val_test[m])
				
		datatest = np.float32( datatest / max_val_test )
		datatest = torch.from_numpy(datatest)
        
    # parameters to train and validate converted to a single precision floating point
	
	print('datatrain shape:', datatrain.shape)
	print('prmtrain shape:', prmtrain.shape)
    
	prmtrain = Tensor(prmtrain[91:,:])
	datatrain = Tensor(datatrain[91:,:])

	print('datatrain :', datatrain.shape)
	print('prmtrain shape:', prmtrain.shape)
    
	combined_tensor = torch.cat((datatrain, prmtrain),dim=1)
	print('combined_tensor: ', combined_tensor.shape)
    
	#datatrain_Class = MyDataset_tensor(combined_tensor, bvals_corrected_train, bvecs_corrected_train)

	### Create mini-batches on training data with data loader.
	loadertrain = DataLoader(combined_tensor, batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses. These lines calculate the number of mini-batches (nobatch) by iterating over the loadertrain. 
	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1
	print('Number of mini-batches created: ', nobatch)	
	print('')

   # It initializes the losstrain and lossval arrays with NaN values to store the losses for each mini-batch and epoch.
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan

	### Instantiate the network and training objects, and save the intantiated network
	nnet = network_cpu_multiple_datasets.mripar(nhidden, pdrop, mrimodel).to('cpu')   # Instantiate neural network in cpu
    
   # Change tissue parameter ranges if specified
	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)  
        
	print('')                           
	print('** Tissue parameter names: {}'.format(nnet.param_name))	
	print('** Tissue parameter lower bounds: {}'.format(nnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(nnet.param_max))	
	print('')
	print('')
    
	nnetloss = nn.MSELoss()                                                         # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                         # Network trained with ADAM optimiser
    
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') )    # Save network at epoch 0 as a state distionary (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()

	### Create normalisation tensors for model parameters. Initialize them with ones. 
   # These arrays will be used for normalization of the model parameters during training.
	slope_norm_tr = np.ones((mbatch , npar_train))
	offset_norm_tr = np.ones((mbatch , npar_train))

   # This loop calculates the normalization factors for each parameter in the model. 
   # It iterates over the range of npar_train (number of trainable parameters) and computes the slope and offset for normalization. 
   # The slope is calculated as the reciprocal of the range of each parameter, and the offset is the negative minimum value divided by the range.
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
	    
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch))
		print('')

		# Loop over mini-batches for at a fixed epoch
		minibatch_id = 0
		for dwi_batch in loadertrain:
			# Pass the mini-batch through the network and store the training loss
			#print('dwi_batch: ', dwi_batch[:,0:nmeas_train])
			output = nnet(dwi_batch[:,0:nmeas_train] )  # Pass MRI measurements and estimate tissue parmaters
			#print('output: ', output)

			try:
                #The training loss is calculated by comparing the estimated tissue parameters (output) with the ground truth tissue parameters
				lossmeas_train = nnetloss(output*slope_norm_tr + offset_norm_tr, dwi_batch[:,nmeas_train:nmeas_train+npar_train]*slope_norm_tr + offset_norm_tr) # Training loss 
			except:
				raise RuntimeError('The number of training voxels must be a multiple of the size of the mini-batch!')

			# Back propagation. Backpropagation is performed by calling losstrain.backward() to compute the gradients and update the network 
            # parameters using nnetopt.step().
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			# Store loss for the current mini-batch of training
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data.detach().clone().cpu())

			# Update mini-batch counter
			minibatch_id = minibatch_id + 1
		
		#print(lossmeas_train.data)
		# Run validation. After processing all the mini-batches for the current epoch, the network is set to evaluation mode
		nnet.eval()   # Set network to evaluation mode (deactivates dropout)
     # The validation dataset (dataval) is passed through the network to obtain the predicted tissue parameters
		tissueval_nnet = nnet( dataval )                  # Output of full network (predicted tissue parameters)
		#print('tissueval_nnet = ', tissueval_nnet)
		#print('\dataval = ', dataval)

     # The predicted tissue parameters are used to estimate MRI signals (dataval_nnet) using the nnet.getsignals() function.
		#[dataval_nnet, g_train, bvals_train] = nnet.getsignals( Tensor(tissueval_nnet) )  # Estimate MRI signals
		dataval_nnet = nnet.getsignals( tissueval_nnet, Tensor(bvals_corrected_val) , Tensor(bvecs_corrected_val) )  # Estimate MRI signals

		#print('dataval_nnet = ', dataval_nnet)

		dataval_nnet = dataval_nnet.detach().cpu().numpy()
		#print('dataval_nnet detatched = ', dataval_nnet)

        # The maximum value of the predicted validation signals (dataval_nnet) along each row is calculated and stored in max_val_val_out.
		max_val_val_out = np.transpose( matlib.repmat(np.max(dataval_nnet,axis=1),nmeas_train,1) )
        
        # The validation loss is computed by comparing the predicted tissue parameters (tissueval_nnet) with the ground truth tissue parameters (prmval).
		lossmeas_val = nnetloss( tissueval_nnet*slope_norm_val + offset_norm_val  , prmval*slope_norm_val + offset_norm_val  ) # Validation loss
		# Store validation loss
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data.detach().clone().cpu())

		# Save trained network at current epoch if validation loss has decreased
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
			loss_val_prev = Tensor.numpy(lossmeas_val.data.detach().clone().cpu())
			# Save predicted validation tissue parameters 
			tissueval_nnet = tissueval_nnet.detach().cpu().numpy()
			#print('tissueval_nnet detatched = ', tissueval_nnet)

			#print('max_val_val = ', max_val_val)
			#print('max_val_val_out = ', max_val_val_out)
			#print('tissueval_nnet = ', tissueval_nnet)
			#print('tissueval_nnet[:,s0idx] = ', tissueval_nnet[:,s0idx])

			tissueval_nnet[:,s0idx] = (max_val_val[:,0]/max_val_val_out[:,0])*tissueval_nnet[:,s0idx] # Rescale s0 (any column of would work)
			#print('tissueval_nnet rescaled = ', tissueval_nnet)

			tissueval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissueval.bin'),'wb')
			pk.dump(tissueval_nnet,tissueval_nnet_file,pk.HIGHEST_PROTOCOL)      
			tissueval_nnet_file.close()
			# Save predicted validation signals

			#print('dataval_nnet = ', tissueval_nnet)
            
			dataval_nnet = (max_val_val/max_val_val_out)*dataval_nnet				
			dataval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()

			# Analyse test data if provided. If test data is provided (args.dtest is not None), the same analysis is performed on 
            # the test data, including obtaining the predicted test tissue parameters (tissuetest_nnet) and the predicted test signals 
            # (datatest_nnet). These are then saved.
			if args.dtest is not None:
				# Get neuronal activations as well as predicted test tissue parameters and test MRI signals 
				#print('datatest = ', datatest)

				tissuetest_nnet = nnet( datatest )              # Output of network (estimated tissue parameters)
                
				datatest_nnet = nnet.getsignals( tissuetest_nnet , Tensor(bvals_corrected_test) , Tensor(bvecs_corrected_test)) # Predicted MRI signals
				datatest_nnet = datatest_nnet.detach().cpu().numpy()
				max_val_test_out = np.transpose( matlib.repmat(np.max(datatest_nnet,axis=1),nmeas_train,1) )
                
				# Save predicted test tissue parameters 
				tissuetest_nnet = tissuetest_nnet.detach().cpu().numpy()
				#print('tissuetest_nnet = ', tissuetest_nnet)
				#print('max_val_test = ', max_val_test)
				#print('max_val_test_out = ', max_val_test_out)

				tissuetest_nnet[:,s0idx] = (max_val_test[:,0]/max_val_test_out[:,0])*tissuetest_nnet[:,s0idx] # Rescale s0 (any column of max_val_test works)
				#print('tissuetest_nnet rescaled = ', tissuetest_nnet)

				tissuetest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissuetest.bin'),'wb')
				pk.dump(tissuetest_nnet,tissuetest_nnet_file,pk.HIGHEST_PROTOCOL)      
				tissuetest_nnet_file.close()
                
				# Save predicted test signals
                
				#print('datatest_nnet = ', datatest_nnet)

				datatest_nnet = (max_val_test/max_val_test_out)*datatest_nnet # Rescale signal	
				#print('datatest_nnet rescaled = ', datatest_nnet)

				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)         
				datatest_nnet_file.close()


		# Set network back to training mode
		nnet.train()

		# Print some information
		print('')
		print('\t\t\t TRAINING INFO:')
		print('\t\t\t Trainig loss: {:.12f}; validation loss: {:.12f}'.format(
			Tensor.numpy(lossmeas_train.data.detach().clone().cpu()), 
			Tensor.numpy(lossmeas_val.data.detach().clone().cpu())) 
		)
		print('')

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
				file.write('MRI protocol (train): ' + '\n')
				file.write('  ** bvals: ' + str(bvals_train_samples.shape) + '\n')
				file.write('  ** bvecs: ' + str(bvals_train_samples.shape) + '\n' + '\n')
                
				file.write('Input signals path: ' + str(args.sig_train) + '\n' + '\n')
            
				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')

