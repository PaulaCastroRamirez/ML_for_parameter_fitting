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
import network_cpu

'''
The output directory will contain:
	 ** losstrain.bin, pickle binary storing the training loss as a numpy matrix (shape: epoch x batch); 
	 ** lossval.bin, pickle binary storing the validation loss as a numpy matrix (shape: epoch x 1); 
	 ** nnet_epoch0.bin, pickle binary storing the qMRI-net at initialisation; 
	 ** nnet_epoch0.pth, Pytorch binary storing the qMRI-net at initialisation; 
	 ** nnet_epoch<FINAL_EPOCH>.bin, pickle binary storing the qMRI-net at the final epoch; 
	 ** nnet_lossvalmin.bin, pickle binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); 
	 ** nnet_lossvalmin.pth, Pytorch binary storing the trained qMRI-net at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); 
	 ** nnet_lossvalmin_sigval.bin, prediction of the validation signals (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); 
	 ** nnet_lossvalmin_tissueval.bin, prediction of tissue parameters from validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); 
	 ** nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss; ** lossval_min.txt, miniimum validation loss; 
	 ** nnet_lossvalmin_sigtest.bin, prediction of the test signals  (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information), if those signals are provided; 
	 ** nnet_lossvalmin_tissuetest.bin, prediction of tissue parameters from test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided')

'''


if __name__ == "__main__":
    
	parser = argparse.ArgumentParser(description='This program trains a network for MRI parameter estimation. Loss calcualted between tissue parameters')
	parser.add_argument('sig_train', 
					 help='path to a pickle binary file storing the input training MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('param_train', 
					 help='path to a pickle binary file storing the training tissue parameter data as a numpy matrix (rows: voxels; columns: parameters)')
	parser.add_argument('sig_val', 
					 help='path to a pickle binary file storing the input validation MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('param_val', 
					 help='path to a pickle binary file storing the validation tissue parameters as a numpy matrix (rows: voxels; columns: parameters)')
	parser.add_argument('mri_model', 
					 help='string indicating the MRI model to fit Zeppelin')
	parser.add_argument('bvals', 
					 help='path to text file storing the bvalues')
	parser.add_argument('bvecs', 
					 help='path to text file storing the gradient directions')
	parser.add_argument('out_base', 
					 help='base name of output directory')
	parser.add_argument('--nn', metavar='<list>', 
					 help='array storing the number of hidden neurons, separated by hyphens (example: 30-15-8)')
	parser.add_argument('--pdrop', metavar='<value>', default='0.0', 
					 help='dropout probability in each layer of the neural network. Default: 0.0')
	parser.add_argument('--noepoch', metavar='<value>', default='100', 
					 help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', 
					 help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', help='number of voxels in each training mini-batch. Default: 1/80 of the total number of training voxels (minimum: 2 voxels)')
	parser.add_argument('--seed', metavar='<value>', default='19102018', 
					 help='integer used as a seed for Numpy and PyTorch random number generators. Default: 19102018')
	parser.add_argument('--nwork', metavar='<value>', default='0', 
					 help='number of workers for data loader. Default: 0')
	parser.add_argument('--dtest', metavar='<file>', 
					 help='path to an option input pickle binary file storing test MRI signals as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('--parmin', metavar='<value>', 
					 help='list of lower bounds of tissue parameters.')
	parser.add_argument('--parmax', metavar='<value>', 
					 help='list of upper bounds of tissue parameters. ')
	args = parser.parse_args()

	### Get some of the inputs
	pdrop = float(args.pdrop)      # dropout probability
	noepoch = int(args.noepoch)    # number of epoch
	lrate = float(args.lrate)      # learning rate
	seed = int(args.seed)          # integer used as a seed for Numpy and PyTorch random number generators
	nwork = int(args.nwork)        # number of workers in dataloader
	mrimodel = args.mri_model      # mri model


	print('\n\nTRAIN A MRI NETWORK (mripar CLASS)                   ')
	print(f'\n** Input training MRI signals: {args.sig_train}')
	print(f'** Input training tissue parameters: {args.param_train}')
	print(f'** Input validation MRI signals: {args.sig_val}')
	print(f'** Input validation tissue parameters: {args.param_val}')
    
	if args.dtest is not None:
		print(f'** Input test MRI signals: {args.dtest}')

	### Load training MRI signals
	fh = open(args.sig_train,'rb')
	datatrain = pk.load(fh)
	fh.close()
    
	nvox_train = datatrain.shape[0]    # number of voxels
	nmeas_train = datatrain.shape[1]   # number of measurements
	print('\nNumber measurements per voxel: ', nmeas_train)
	print('Number of training voxels: ', nvox_train)

	### Load validation MRI signals
	fh = open(args.sig_val,'rb')
	dataval = pk.load(fh)
	fh.close()
	nvox_val = dataval.shape[0]
	
	print('Number of validation voxels: ', nvox_val)

    
	if dataval.shape[1]!=datatrain.shape[1]:
		raise RuntimeError('the number of MRI measurements in the validation set differs from the training set!')		

	### Load test MRI signals
	if args.dtest is not None:
		fh = open(args.dtest,'rb')
		datatest = np.float32(pk.load(fh))
		fh.close()
		
		nvox_test = datatest.shape[0]
		print('Number of test voxels: ', nvox_test)
		
		if datatest.shape[1]!=datatrain.shape[1]:
			raise RuntimeError('the number of MRI measurements in the test set differs from the training set!')		


	### Load training tissue parameters
	fh = open(args.param_train,'rb')
	prmtrain = pk.load(fh)
	npar_train = prmtrain.shape[1]
	fh.close()
	
	if prmtrain.shape[0]!=datatrain.shape[0]:
		raise RuntimeError('the number of voxels in the training parameters differs from the training MRI signals!')		

	### Load validation tissue parameters
	fh = open(args.param_val,'rb')
	prmval = pk.load(fh)
	fh.close()
	
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

	### Load MRI protocol. bvals and bvecs
	try:
        
		#bvals = np.loadtxt(args.bvals)
		fh = open(args.bvals,'rb')
		bvals = pk.load(fh)
		fh.close()
		print(bvals.shape)
		
		#g = np.loadtxt(args.bvecs)
		fh = open(args.bvecs,'rb')
		g = pk.load(fh)
		fh.close()
		print(g.shape)
                           
	except: 		
		raise RuntimeError('the MRI protocol is not in a suitable text file format. Sorry!')

	bvals = np.array(bvals)/1000.0
	bvals = bvals[:108]
	g = g[:108,:]

	print(bvals.shape)
	print(g.shape)
	
	Nmeas = bvals.size
    
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
	out_base_dir = '{}_nhidden{}_pdrop{}_noepoch{}_lr{}_mbatch{}_seed{}'.format(args.out_base,nhidden_str,pdrop,noepoch,lrate,mbatch,seed)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir) # create the folder if it does not exist

	print(f'** Output directory: {out_base_dir}')
	print('\n\nPARAMETERS')
	print(f'\n** Hidden neurons: {nhidden}')
	print(f'** Dropout probability: {pdrop}')
	print(f'** Number of epochs: {noepoch}')
	print(f'** Learning rate: {lrate}')
	print(f'** Number of voxels in a mini-batch: {mbatch}')
	print(f'** Seed: {seed}')
	print(f'** Number of workers for data loader: {nwork} \n\n')


	### Set random seeds.
	np.random.seed(seed)

	### Normalise MRI signals and convert to single precision. Normalize the datatrain array by dividing it by the maximum value along each row. It performs element-wise division to ensure that the data is in the range [0, 1].
	max_val_train = np.transpose( matlib.repmat(np.max(datatrain,axis=1),nmeas_train,1) )
	for m, i in enumerate(max_val_train):
        	if i.any() == 0:
        		max_val_train[m]= np.ones_like(max_val_train[m])
	datatrain = np.float32( datatrain / max_val_train )
    

	max_val_train_param = np.max(prmtrain[:, s0idx])
	prmtrain[:,s0idx] = np.float32( prmtrain[:,s0idx] / max_val_train_param )
        
	max_val_val = np.transpose( matlib.repmat(np.max(dataval,axis=1),nmeas_train,1) ) 
	for m, i in enumerate(max_val_val):
        	if i.any() == 0:
        		max_val_val[m]= np.ones_like(max_val_val[m])
				
	dataval = np.float32( dataval / max_val_val )
	dataval = torch.from_numpy(dataval)
    
	max_val_val_param = np.max(prmval[:, s0idx])
	prmval[:,s0idx] = np.float32( prmval[:,s0idx] / max_val_val_param )
    
	if args.dtest is not None:
		max_val_test = np.transpose( matlib.repmat(np.max(datatest,axis=1),nmeas_train,1) )
		for m, i in enumerate(max_val_test):
			if i.any() == 0:
				max_val_test[m]= np.ones_like(max_val_test[m])
				
		datatest = np.float32( datatest / max_val_test )
		datatest = torch.from_numpy(datatest)
        
    # parameters to train and validate converted to a single precision floating point
	prmtrain = np.float32(prmtrain)
	prmval = np.float32(prmval)
	prmval = torch.from_numpy(prmval)
	
	prmtrain = Tensor(prmtrain)
	datatrain = Tensor(datatrain)
    
	### Create mini-batches on training data with data loader.
	loadertrain = DataLoader(torch.cat((datatrain, prmtrain),dim=1), batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses. 
	nobatch=0
	for signals in loadertrain:
		nobatch = nobatch+1
	print('Number of mini-batches created: ', nobatch, '\n')	

	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan

	nnet = network_cpu.mripar(nhidden, pdrop, mrimodel, bvals, g).to('cpu')     # Instantiate neural network in cpu
    
   # Change tissue parameter ranges if specified
	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)  
        
	print(f'\n** Tissue parameter names: {nnet.param_name}')	
	print(f'** Tissue parameter lower bounds: {nnet.param_min}')	
	print(f'** Tissue parameter upper bounds: {nnet.param_max} \n\n')	

	nnetloss = nn.MSELoss()                                                         # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                         # Network trained with ADAM optimiser
    
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') )    # Save network at epoch 0 as a state distionary (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()

	# Create normalisation tensors for model parameters. 
	slope_norm_tr = np.ones((mbatch , npar_train))
	offset_norm_tr = np.ones((mbatch , npar_train))

    #normalization factors for each parameter in the model. 
	for pp in range(0,npar_train):
		slope_norm_tr[:,pp] = 1.0 / (nnet.param_max[pp] - nnet.param_min[pp])
		offset_norm_tr[:,pp] = (-1.0*nnet.param_min[pp]) / (nnet.param_max[pp] - nnet.param_min[pp])

	slope_norm_tr = Tensor(np.float32(slope_norm_tr))
	offset_norm_tr = Tensor(np.float32(offset_norm_tr))

	slope_norm_val = np.ones((nvox_val , npar_train))
	offset_norm_val = np.ones((nvox_val , npar_train))

	for pp in range(0,npar_train):
		slope_norm_val[:,pp] = 1.0 / (nnet.param_max[pp] - nnet.param_min[pp])
		offset_norm_val[:,pp] = (-1.0*nnet.param_min[pp]) / (nnet.param_max[pp] - nnet.param_min[pp])

	slope_norm_val = Tensor(np.float32(slope_norm_val))
	offset_norm_val = Tensor(np.float32(offset_norm_val))
    

	###################### Run training ###################################
    
	start_time = time.time()

	loss_val_prev = np.inf
	for epoch in range(noepoch):
	    
		print(f'        EPOCH   {epoch+1}/{noepoch} \n')

		minibatch_id = 0
		for signals in loadertrain:
			
			output = nnet(signals[:,0:nmeas_train] )  # Pass MRI measurements and estimate tissue parmaters

			try:
				lossmeas_train = nnetloss(output*slope_norm_tr + offset_norm_tr, signals[:,nmeas_train:nmeas_train+npar_train]*slope_norm_tr + offset_norm_tr) # Training loss 
			except:
				raise RuntimeError('The number of training voxels must be a multiple of the size of the mini-batch!')

			# Back propagation.
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data.detach().clone().cpu())

			minibatch_id = minibatch_id + 1
		
		nnet.eval()   # evaluation mode (deactivates dropout)
		tissueval_nnet = nnet( dataval )                  # predicted val parameters

		dataval_nnet = nnet.getsignals( tissueval_nnet )  # Estimatef val MRI signals
		dataval_nnet = dataval_nnet.detach().cpu().numpy()

		max_val_val_out = np.transpose( matlib.repmat(np.max(dataval_nnet,axis=1),nmeas_train,1) ) # rescaling factor for signals
        
		lossmeas_val = nnetloss( tissueval_nnet*slope_norm_val + offset_norm_val  , prmval*slope_norm_val + offset_norm_val  ) # Validation loss
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
			nnet_text.write('Epoch {} (indices starting from 0)'.format(epoch))
			nnet_text.close()
			
			# Update value of best validation loss
			loss_val_prev = Tensor.numpy(lossmeas_val.data.detach().clone().cpu())
			tissueval_nnet = tissueval_nnet.detach().cpu().numpy()

			tissueval_nnet[:,s0idx] = (max_val_val[:,0]/max_val_val_out[:,0])*tissueval_nnet[:,s0idx] # Rescale s0 

			tissueval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissueval.bin'),'wb')
			pk.dump(tissueval_nnet,tissueval_nnet_file,pk.HIGHEST_PROTOCOL)      
			tissueval_nnet_file.close()
            
			dataval_nnet = (max_val_val/max_val_val_out)*dataval_nnet				
			dataval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()

			# Analyse test data if provided
			if args.dtest is not None:
				tissuetest_nnet = nnet( datatest )                 # estimated test tissue parameters
                
				datatest_nnet = nnet.getsignals( tissuetest_nnet ) # Predicted  test MRI signals
				datatest_nnet = datatest_nnet.detach().cpu().numpy()
				max_val_test_out = np.transpose( matlib.repmat(np.max(datatest_nnet,axis=1),nmeas_train,1) )
                
				tissuetest_nnet = tissuetest_nnet.detach().cpu().numpy()
				tissuetest_nnet[:,s0idx] = (max_val_test[:,0]/max_val_test_out[:,0])*tissuetest_nnet[:,s0idx] # Rescale s0 
				
				tissuetest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_tissuetest.bin'),'wb')
				pk.dump(tissuetest_nnet,tissuetest_nnet_file,pk.HIGHEST_PROTOCOL)      
				tissuetest_nnet_file.close()
                
				datatest_nnet = (max_val_test/max_val_test_out)*datatest_nnet # Rescale signal	

				datatest_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigtest.bin'),'wb')
				pk.dump(datatest_nnet,datatest_nnet_file,pk.HIGHEST_PROTOCOL)         
				datatest_nnet_file.close()


		nnet.train()

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
				file.write('MRI protocol: ' + '\n')
				file.write('  ** bvals: ' + str(bvals.shape) + '\n')
				file.write('  ** bvecs: ' + str(g.shape) + '\n')
				file.write('Input signals path: ' + str(args.sig_train) + '\n' + '\n')
            
				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')

