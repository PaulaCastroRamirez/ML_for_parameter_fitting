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

# ## Load libraries

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
#import network_cpu
#import network_cpu_multiple_datasets_single_pixel
import network_cpu_multiple_datasets
#import network_cpu_multiple_datasets_prueba2
#import plotext

from my_dataset import MyDataset
import time


if __name__ == "__main__":
    

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program trains a qMRI-net for quantitative MRI parameter estimation. A qMRI-Nnet enables voxel-by-voxel estimation of microstructural properties from sets of MRI images aacquired by varying the MRI sequence parameters.  Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('dtrain', help='path to a pickle binary file storing the input training data as a numpy matrix (rows: voxels; columns: measurements)')
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
	#print('** Input validation data: {}'.format(args.dval))
	if args.dtest is not None:
		print('** Input test data: {}'.format(args.dtest))

    # Paths
	train_path = args.dtrain
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
	bvals_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas ))
	bvecs_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas, 3))
	msk_train_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files)))

# =============================================================================
# 	print('dwi_train_: ', dwi_train_.shape)
# 	print('grad_dev_train_: ', grad_dev_train_.shape)
# 	print('bvals_train_: ', bvals_train_.shape)
# 	print('bvecs_train_: ', bvecs_train_.shape)
# 	print('bvecs_train_: ', msk_train_.shape)
# =============================================================================

	slices2 = 0
    
	# Iterate over the files and load them in one single array
	for i, file in enumerate(dwi_files):
    
		print('')
		print('Num dwi file: ', i)

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
         
		slices1 = slices2
		slices2 = dwi_train.shape[2] + slices1
    
		print('From Slice: ', slices1, ' To slice: ', slices2)
    
		bvals_repeated = np.tile(bvals_train, (x_shape, y_shape, 1, 1))
		bvecs_repeated = np.tile(bvecs_train, (x_shape, y_shape, 1, 1, 1))
		print('bvals_repeated: ', bvals_repeated.shape)
		print('bvecs_repeated: ', bvecs_repeated.shape)

		dwi_train_[:,:,slices1:slices2,:]       = dwi_train
		grad_dev_train_[:,:,slices1:slices2,:]  = grad_dev_train
		bvals_train_[:,:,slices1:slices2, :]    = bvals_train
		bvecs_train_[:,:,slices1:slices2, :,:]  = bvecs_train
		msk_train_[:,:,slices1:slices2]         = msk_train


# =============================================================================
	print('')
	print('dwi_train_: ', dwi_train_.shape)
	print('grad_dev_train_: ', grad_dev_train_.shape)
	print('bvals_train_: ', bvals_train_.shape)
	print('bvecs_train_: ', bvecs_train_.shape)
	print('msk_train_: ', msk_train_.shape)
# =============================================================================

	print('')
	dwi_train_samples_together = dwi_train_[msk_train_ == 1]
	print('dwi_train_samples_together: ', dwi_train_samples_together.shape)

	grad_dev_train_samples_together = grad_dev_train_[msk_train_ == 1]
	print('grad_dev_train_samples_together: ', grad_dev_train_samples_together.shape)

	bvals_train_samples_together = bvals_train_[msk_train_ == 1]
	print('bvals_train_samples_together: ', bvals_train_samples_together.shape)
    
	bvecs_train_samples_together = bvecs_train_[msk_train_ == 1]
	print('bvecs_train_samples_together: ', bvecs_train_samples_together.shape)

	if args.perVAL is None:
		percentage = args.perVal
	else:
		percentage = 0.15
        

	# Use the indices to extract the corresponding subsets of the data
	dwi_train_samples = dwi_train_samples_together[:2068397, :]
	grad_dev_train_samples = grad_dev_train_samples_together[:2068397, :]
	bvals_train_samples = bvals_train_samples_together[:2068397, :]
	bvecs_train_samples = bvecs_train_samples_together[:2068397, :, :]

	dwi_val_samples = dwi_train_samples_together[2068397:, :]
	grad_dev_val_samples = grad_dev_train_samples_together[2068397:, :]
	bvals_val_samples = bvals_train_samples_together[2068397:, :]
	bvecs_val_samples = bvecs_train_samples_together[2068397:, :, :]

	print('')
	print('**** SIZES TRAIN AND VALIDATION SAMPLES ***')
	print('')

	print('dwi_train_samples: ', dwi_train_samples.shape)
	print('grad_dev_train_samples: ', grad_dev_train_samples.shape)
	print('bvals_train_samples: ', bvals_train_samples.shape)
	print('bvecs_train_samples: ', bvecs_train_samples.shape)
	print('')
    
	print('dwi_val_samples: ', dwi_val_samples.shape)
	print('grad_dev_val_samples: ', grad_dev_val_samples.shape)
	print('bvals_val_samples: ', bvals_val_samples.shape)
	print('bvecs_val_samples: ', bvecs_val_samples.shape)

    
	print('')
	print('... pre-calculating bvals and bvecs corrected TRAINING ')
	print('')

	bvals_corrected_train = np.zeros((bvals_train_samples.shape))
	bvecs_corrected_train = np.zeros((bvecs_train_samples.shape))

	start_time_train = time.time()
	for ii in range(0,dwi_train_samples.shape[0],1):
		L = np.array([[grad_dev_train_samples[ii,0], grad_dev_train_samples[ii,1], grad_dev_train_samples[ii,2]],
                    [grad_dev_train_samples[ii,3], grad_dev_train_samples[ii,4], grad_dev_train_samples[ii,5]],
                    [grad_dev_train_samples[ii,6], grad_dev_train_samples[ii,7], grad_dev_train_samples[ii,8]]])

   	 	# Apply gradient non-linearities correction
		I = np.eye(3)                          # identity matrix               
		v = np.dot(bvecs_train_samples[ii,:,:], (I + L))
		n = np.sqrt(np.diag(np.dot(v, v.T)))

		new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
		new_bval = n ** 2 * bvals_train_samples[ii,:]
		new_bvec[new_bval == 0, :] = 0
		bvals_corrected_train[ii,:] = Tensor(new_bval)
		bvecs_corrected_train[ii,:,:] = Tensor(new_bvec)
    
	end_time_train = time.time()
	print('Time to pre-compute bvals and bvecs corrected train: ', end_time_train - start_time_train)
    
	print('')
	print('... pre-calculating bvals and bvecs corrected VALIDATION ')
	print('')

	bvals_corrected_val = np.zeros((bvals_val_samples.shape))
	bvecs_corrected_val = np.zeros((bvecs_val_samples.shape))

	start_time_val = time.time()
	for ii in range(0,dwi_val_samples.shape[0],1):
		L = np.array([[grad_dev_val_samples[ii,0], grad_dev_val_samples[ii,1], grad_dev_val_samples[ii,2]],
                    [grad_dev_val_samples[ii,3], grad_dev_val_samples[ii,4], grad_dev_val_samples[ii,5]],
                    [grad_dev_val_samples[ii,6], grad_dev_val_samples[ii,7], grad_dev_val_samples[ii,8]]])

   	 	# Apply gradient non-linearities correction
		I = np.eye(3)                          # identity matrix               
		v = np.dot(bvecs_val_samples[ii,:,:], (I + L))
		n = np.sqrt(np.diag(np.dot(v, v.T)))

		new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
		new_bval = n ** 2 * bvals_val_samples[ii,:]
		new_bvec[new_bval == 0, :] = 0
		bvals_corrected_val[ii,:] = Tensor(new_bval)
		bvecs_corrected_val[ii,:,:] = Tensor(new_bvec)
    
	end_time_val = time.time()
	print('Time to pre-compute bvals and bvecs corrected val: ', end_time_val - start_time_val)

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
		grad_dev_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), 9))
		bvals_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas ))
		bvecs_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files), N_meas, 3))
		msk_test_ = np.zeros((x_shape,y_shape,num_slices*len(dwi_files)))

		print('')
		print('dwi_test_: ', dwi_test_.shape)
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
			grad_dev_test_[:,:,slices1:slices2,:]  = grad_dev_test
			bvals_test_[:,:,slices1:slices2, :]    = bvals_test
			bvecs_test_[:,:,slices1:slices2, :,:]  = bvecs_test
			msk_test_[:,:,slices1:slices2]         = msk_test

# =============================================================================
		print('')
		print('dwi_test_: ', dwi_test_.shape)
		print('grad_dev_test_: ', grad_dev_test_.shape)
		print('bvals_test_: ', bvals_test_.shape)
		print('bvecs_test_: ', bvecs_test_.shape)
		print('msk_test_: ', msk_test_.shape)
# =============================================================================

		print('')
		dwi_test_samples = dwi_test_[msk_test_ == 1]
		print('dwi_test_samples: ', dwi_test_samples.shape)

		grad_dev_test_samples = grad_dev_test_[msk_test_ == 1]
		print('grad_dev_test_samples: ', grad_dev_test_samples.shape)

		bvals_test_samples = bvals_test_[msk_test_ == 1]
		print('bvals_test_samples: ', bvals_test_samples.shape)

		bvecs_test_samples = bvecs_test_[msk_test_ == 1]
		print('bvecs_test_samples: ', bvecs_test_samples.shape)
        
		print('')
		print('... pre-calculating bvals and bvecs corrected TEST ')
		print('')

		bvals_corrected_test = np.zeros((bvals_test_samples.shape))
		bvecs_corrected_test = np.zeros((bvecs_test_samples.shape))

		start_time_test = time.time()
		for ii in range(0,dwi_test_samples.shape[0],1):
			L = np.array([[grad_dev_test_samples[ii,0], grad_dev_test_samples[ii,1], grad_dev_test_samples[ii,2]],
                    [grad_dev_test_samples[ii,3], grad_dev_test_samples[ii,4], grad_dev_test_samples[ii,5]],
                    [grad_dev_test_samples[ii,6], grad_dev_test_samples[ii,7], grad_dev_test_samples[ii,8]]])

   	 		# Apply gradient non-linearities correction
			I = np.eye(3)                          # identity matrix               
			v = np.dot(bvecs_test_samples[ii,:,:], (I + L))
			n = np.sqrt(np.diag(np.dot(v, v.T)))

			new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
			new_bval = n ** 2 * bvals_test_samples[ii,:]
			new_bvec[new_bval == 0, :] = 0
			bvals_corrected_test[ii,:] = Tensor(new_bval)
			bvecs_corrected_test[ii,:,:] = Tensor(new_bvec)
    
		end_time_test = time.time()

		print('Time to pre-compute bvals and bvecs corrected test: ', end_time_test - start_time_test)


	print('')

	### Get number of mini-batches
	if args.mbatch is None:
		mbatch = int(float(dwi_train_samples.shape[0]) / 80.0) # Default: 1/80 of the total number of training voxels
		print('dwi_train_samples.shape[0]: ', dwi_train_samples.shape[0])
		print('mbatch: ', mbatch)
	else:
		mbatch = int(args.mbatch)
		if (mbatch>dwi_train_samples.shape[0]):
			mbatch = dwi_train_samples.shape[0]
		if(mbatch<2):
			mbatch = int(2)


	nmeas_train = dwi_train_samples.shape[1]
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


	### Create output base name
	out_base_dir = '{}_nhidden{}_pdrop{}_noepoch{}_lr{}_mbatch{}_seed{}'.format(args.out_base,nhidden_str,pdrop,noepoch,lrate,mbatch,seed)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir)

	### Print some more information
	print('')
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

	### Set random seeds
	np.random.seed(seed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(seed)    # Random seed for reproducibility: PyTorch

	### Normalise data
	max_val_train = np.transpose( matlib.repmat(np.max(dwi_train_samples,axis=1), nmeas_train,1) )
	for m, i in enumerate(max_val_train):
        	if i.any() == 0:
        		#print('original:', m)
        		max_val_train[m]= np.ones_like(max_val_train[m])
        		#print('') 
        		#print('new: ', max_val_train[m])

	datatrain = np.float32( dwi_train_samples / max_val_train )
	#print('data_train:', datatrain[88493,:].shape)

    # normalize the dataval array using the maximum values along each row.
	max_val_val = np.transpose( matlib.repmat(np.max(dwi_val_samples,axis=1),nmeas_train,1) )
	for m, i in enumerate(max_val_val):
        	if i.any() == 0:
        		max_val_val[m]= np.ones_like(max_val_val[m])

	dataval = np.float32( dwi_val_samples / max_val_val )

    # If it is not None, it means there is test data available, and the code normalizes the datatest array using the maximum values along each row.
	if args.dtest is not None:
		max_val_test = np.transpose( matlib.repmat(np.max(dwi_test_samples,axis=1),nmeas_train,1) )
		for m, i in enumerate(max_val_test):
			if i.any() == 0:
				max_val_test[m]= np.ones_like(max_val_test[m])

		datatest = np.float32( dwi_test_samples / max_val_test )	

    
	print('datatrain shape: ', datatrain.shape)
	print('dataval shape: ', dataval.shape)
	print('datatest shape: ', datatest.shape)

    # create an instance of the DATASET class containing the trianing data: pixel vlaues of the dwi image, bvals and bvecs
	datatrain_Class = MyDataset(datatrain, bvals_corrected_train, bvecs_corrected_train)
	loadertrain = DataLoader(datatrain_Class, batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses
	nobatch=0   # Count how many mini-batches of size mbatch we created
	for signals in loadertrain:
		nobatch = nobatch+1

	print('')
	print('Number of mini-batches created: ', nobatch)
	print('')
    
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan
    
	### Instantiate the network and training objects, and save the intantiated network
	nnet = network_cpu_multiple_datasets.mrisig(nhidden, pdrop, mrimodel).to('cpu')    # Instantiate neural network

	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)       

	print('')                           
    # Change tissue parameter ranges
	print('** Tissue parameter names: {}'.format(nnet.param_name))
	print('** Tissue parameter lower bounds: {}'.format(nnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(nnet.param_max))	
	print('')
	print('')	
	nnetloss = nn.MSELoss()                                                      # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                      # Network trained with ADAM optimiser
    
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') )    # Save network at epoch 0 (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()


	###################### Run training ###################################

	# Loop over epochs
	start_time = time.time()
	loss_val_prev = np.inf
	for epoch in range(noepoch):
		numk = 0
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch))
		print('')

		# Loop over mini-batches for at a fixed epoch
		minibatch_id = 0
		for batch in loadertrain:
            
			dwi_batch = batch['dwi']
			bvals_batch = batch['bvals']
			bvecs_batch = batch['bvecs']
                
			#print('')
			#print('in ', dwi_batch[50,:].shape)
			#print('in ', dwi_batch[50,:])

			#print(bvals_batch[50,:])
			#print(bvecs_batch[50,:,:])
			#to_plot = dwi_batch[50,:]
			#plotext.plot(to_plot)
			#plotext.title('input')
			#plotext.show()
# =============================================================================
# 			print(bvals_batch.shape)
# 			print(bvecs_batch.shape)
# =============================================================================

			# Pass the mini-batch through the network and store the training loss
			output = nnet(Tensor(dwi_batch), Tensor(bvals_batch), Tensor(bvecs_batch))      # Pass MRI measurements through net and get estimates of MRI signals  
			#print('out ', output[50,:].shape)
			#print('out ', output[50,:])
            
			#to_plot = output[50,:]
			#plotext.plot(to_plot)
			#plotext.title('output')
			#plotext.show()
            
			#numk = numk+1
			#print('batch: ', numk, '/', nobatch )

			lossmeas_train = nnetloss(output, dwi_batch)  # Training loss
            
			# Back propagation
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			# Store loss for the current mini-batch of training
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data)
			# Update mini-batch counter
			minibatch_id = minibatch_id + 1
  

		# Run validation

		nnet.eval()   # Set network to evaluation mode (deactivates dropout)
		dataval_nnet = nnet( Tensor(dataval) , Tensor(bvals_corrected_val) , Tensor(bvecs_corrected_val) )                     # Output of full network (predicted MRI signals)

		tissueval_nnet = nnet.getparams( Tensor(dataval) )        # Estimated tissue parameters
		neuronval_nnet = nnet.getneurons( Tensor(dataval) )       # Output neuron activations
        
		lossmeas_val = nnetloss( dataval_nnet  , Tensor(dataval)  ) # Validation loss

		dataval_nnet = dataval_nnet.detach().numpy()

        # The maximum value of the predicted validation signals (dataval_nnet) along each row is calculated and stored in max_val_val_out.
		max_val_val_out = np.transpose( matlib.repmat(np.max(dataval_nnet,axis=1),nmeas_train,1) )

		# Store validation loss
		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data)


		# Save trained network at current epoch if validation loss has decreased
		if(Tensor.numpy(lossmeas_val.data)<=loss_val_prev):

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

			# Save predicted validation signals
			dataval_nnet = (max_val_val/max_val_val_out)*dataval_nnet   # Rescale signals
			dataval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()
            
			# Save predicted validation tissue parameters 
			tissueval_nnet = tissueval_nnet.detach().numpy()
			#print('tissueval_nnet detatched = ', tissueval_nnet)
            
			tissueval_nnet[:,s0idx] = (max_val_val[:,0]/max_val_val_out[:,0])*tissueval_nnet[:,s0idx] # # Rescale s0 (any column of would work)
			#print('tissueval_nnet[:,s0idx] = ', tissueval_nnet[:,s0idx])

			tissueval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissueval.bin'),'wb')
			pk.dump(tissueval_nnet,tissueval_nnet_file,pk.HIGHEST_PROTOCOL)      
			tissueval_nnet_file.close()
            
			# Save output validation neuron activations 
			neuronval_nnet = neuronval_nnet.detach().numpy()
			neuronval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_neuronval.bin'),'wb')
			pk.dump(neuronval_nnet,neuronval_nnet_file,pk.HIGHEST_PROTOCOL)      
			neuronval_nnet_file.close()

			# Analyse test data if provided
			if args.dtest is not None:
				# Get neuronal activations as well as predicted test tissue parameters and test MRI signals 

				datatest_nnet = nnet( Tensor(datatest) , Tensor(bvals_corrected_test) , Tensor(bvecs_corrected_test)  )              # Output of full network (predicted MRI signals)
				datatest_nnet = datatest_nnet.detach().numpy()
                
				max_val_test_out = np.transpose( matlib.repmat(np.max(datatest_nnet,axis=1),nmeas_train,1) )
				tissuetest_nnet = nnet.getparams( Tensor(datatest) )  # Estimated tissue parameters
				neurontest_nnet = nnet.getneurons( Tensor(datatest)  ) # Output neuron activations
                
                
				# Save predicted test signals
				datatest_nnet = (max_val_test/max_val_test_out)*datatest_nnet # Rescale signals
				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)      
				datatest_nnet_file.close()
                
				# Save predicted test parameters 
				tissuetest_nnet = tissuetest_nnet.detach().numpy()
				tissuetest_nnet[:,s0idx] = (max_val_test[:,0]/max_val_test_out[:,0])*tissuetest_nnet[:,s0idx] # Rescale s0 (any column works)
				tissuetest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissuetest.bin'),'wb')
				pk.dump(tissuetest_nnet,tissuetest_nnet_file,pk.HIGHEST_PROTOCOL)        
				tissuetest_nnet_file.close()
                
				# Save output test neuron activations 
				neurontest_nnet = neurontest_nnet.detach().numpy()
				neurontest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_neurontest.bin'),'wb')
				pk.dump(neurontest_nnet,neurontest_nnet_file,pk.HIGHEST_PROTOCOL)      
				neurontest_nnet_file.close()

		# Set network back to training mode
		nnet.train()

		# Print some information
		print('')
		print('             TRAINING INFO:')
		print('             Training loss: {:.12f}; validation loss: {:.12f}'.format(Tensor.numpy(lossmeas_train.data), Tensor.numpy(lossmeas_val.data)) )
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
                
				file.write('Input measurements path training: ' + str(args.dtrain) + '\n' + '\n')
				#file.write('Input measurements path validation: ' + str(args.dval) + '\n' + '\n')

				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')
