# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:49:54 2023

@author: pcastror
"""


### Load libraries
import argparse, os, sys
from numpy import matlib
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import pickle as pk
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import network_cpu_single_bmatrix

import time

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
	** nnet_lossvalmin_neuronval.bin, output neuron activations for validation signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information); 
	** nnet_lossvalmin.info, text file reporting information regarding the epoch with the lowest validation loss; 
	** lossval_min.txt, miniimum validation loss;  
	** nnet_lossvalmin_sigtest.bin, prediction of the test signals  (shape: voxels x measurements) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information), if those signals are provided; 
	** nnet_lossvalmin_tissuetest.bin, prediction of tissue parameters from test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided; 
	** nnet_lossvalmin_neurontest.bin, output neuron activations for test signals (shape: voxels x number_of_tissue_parameters) at the best epoch (epoch with lowest validation loss, check nnet_lossvalmin.info file for more information) if test signals are provided')
'''

if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program trains a MRI network for MRI parameter estimation. Loss calcualted between signals')
	parser.add_argument('data_train', 
					 help='path to a pickle binary file storing the input training data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('data_val', 
					 help='path to a pickle binary file storing the validation data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit. Only Zeppelin implemented')
	parser.add_argument('bvals', 
					 help='path to text file storing the bvalues')
	parser.add_argument('bvecs', 
					 help='path to text file storing the gradient directions')
	parser.add_argument('out_base', 
					 help='base name of output directory.')
	parser.add_argument('--nn', metavar='<list>', 
					 help='array storing the number of hidden neurons, separated by hyphens (example: 30-15-8).')
	parser.add_argument('--pdrop', metavar='<value>', default='0.0', 
					 help='dropout probability in each layer of the neural network. Default: 0.0')
	parser.add_argument('--noepoch', metavar='<value>', default='500', 
					 help='number of epochs used for training. Default: 500')
	parser.add_argument('--lrate', metavar='<value>', default='0.001', 
					 help='learning rate. Default: 0.001')
	parser.add_argument('--mbatch', metavar='<value>', 
					 help='number of voxels in each training mini-batch. Default: 1/80 of the total number of training voxels (minimum: 2 voxels)')
	parser.add_argument('--seed', metavar='<value>', default='19102018', 
					 help='integer used as a seed for Numpy and PyTorch random number generators. Default: 19102018')
	parser.add_argument('--nwork', metavar='<value>', default='0', 
					 help='number of workers for data loader. Default: 0')
	parser.add_argument('--dtest', metavar='<file>', 
					 help='path to an option input pickle binary file storing the test data as a numpy matrix (rows: voxels; columns: measurements)')
	parser.add_argument('--parmin', metavar='<value>', 
					 help='list of lower bounds of tissue parameters')
	parser.add_argument('--parmax', metavar='<value>', 
					 help='list of upper bounds of tissue parameters.')
	args = parser.parse_args()

	### Get some of the inputs
	pdrop = float(args.pdrop)
	noepoch = int(args.noepoch)
	lrate = float(args.lrate)
	seed = int(args.seed)
	nwork = int(args.nwork)
	mrimodel = args.mri_model

	print('\n\nTRAIN A MRI NETWORK (mripar CLASS)                   ')
	print(f'\n** Input training MRI signals: {args.data_train}')
	print(f'** Input validation MRI signals: {args.data_val}')
	if args.dtest is not None:
		print(f'** Input test MRI signals: {args.dtest}')

	### Load training data
	fh = open(args.data_train,'rb')
	datatrain = np.float32(pk.load(fh))
	nmeas_train = datatrain.shape[1]
	fh.close()
	
	print('\nNumber measurements per voxel: ', nmeas_train)

	### Load validation data
	fh = open(args.data_val,'rb')
	dataval = np.float32(pk.load(fh))
	fh.close()
	if dataval.shape[1]!=datatrain.shape[1]:
		raise RuntimeError('the number of MRI measurements in the validation set differs from the training set!')		

	### Load test MRI signals
	if args.dtest is not None:
		fh = open(args.dtest,'rb')
		datatest = np.float32(pk.load(fh))
		fh.close()
		if datatest.shape[1]!=datatrain.shape[1]:
			raise RuntimeError('the number of MRI measurements in the test set differs from the training set!')		

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


	### Load MRI protocol
	try:
        
		bvals = np.loadtxt(args.bvals)
		print(bvals.shape)
		
		g = np.loadtxt(args.bvecs)
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
		s0idx = 4


	### Get specifics for hidden layers
	if args.nn is None:

		if (mrimodel=='Zeppelin'): 
			npars = 5                        # number of parameters different in each model
		else:
			raise RuntimeError('the chosen MRI model is not implemented. Sorry!')

        # number og hidden neurons
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

	print(f'** Output directory: {out_base_dir}')
	print('\n\nPARAMETERS')
	print(f'\n** Hidden neurons: {nhidden}')
	print(f'** Dropout probability: {pdrop}')
	print(f'** Number of epochs: {noepoch}')
	print(f'** Learning rate: {lrate}')
	print(f'** Number of voxels in a mini-batch: {mbatch}')
	print(f'** Seed: {seed}')
	print(f'** Number of workers for data loader: {nwork} \n\n')

	### Set random seeds
	np.random.seed(seed)

	### Normalise data
	max_val_train = np.transpose( matlib.repmat(np.max(datatrain,axis=1),nmeas_train,1) )
	for m, i in enumerate(max_val_train):
        	if i.any() == 0:
        		max_val_train[m]= np.ones_like(max_val_train[m])

	datatrain = np.float32( datatrain / max_val_train )

	max_val_val = np.transpose( matlib.repmat(np.max(dataval,axis=1),nmeas_train,1) )
	for m, i in enumerate(max_val_val):
        	if i.any() == 0:
        		max_val_val[m]= np.ones_like(max_val_val[m])

	dataval = np.float32( dataval / max_val_val )

	if args.dtest is not None:
		max_val_test = np.transpose( matlib.repmat(np.max(datatest,axis=1),nmeas_train,1) )
		for m, i in enumerate(max_val_test):
			if i.any() == 0:
				max_val_test[m]= np.ones_like(max_val_test[m])

		datatest = np.float32( datatest / max_val_test )	


	### Create mini-batches on training data with data loader
	loadertrain = DataLoader(datatrain, batch_size=mbatch, shuffle=True, num_workers=nwork)

	### Allocate memory for losses
	nobatch=0  
	for signals in loadertrain:
		nobatch = nobatch+1
	print('Number of mini-batches created: ', nobatch, '\n')	
    
	losstrain = np.zeros((noepoch,nobatch)) + np.nan
	lossval = np.zeros((noepoch,1)) + np.nan
    

	### Instantiate the network and training objects, and save the intantiated network
	nnet = network_cpu_single_bmatrix.mrisig(nhidden, pdrop, mrimodel, bvals, g).to('cpu')                # Instantiate neural network
	if (args.parmin is not None) or (args.parmax is not None):
		nnet.changelim(pminbound,pmaxbound)       

	print(f'\n** Tissue parameter names: {nnet.param_name}')	
	print(f'** Tissue parameter lower bounds: {nnet.param_min}')	
	print(f'** Tissue parameter upper bounds: {nnet.param_max} \n\n')
	
	nnetloss = nn.MSELoss()                                                         # Loss: L2 norm (mean squared error, Gaussian noise)
	nnetopt = torch.optim.Adam(nnet.parameters(), lr=lrate)                         # Network trained with ADAM optimiser
    
	torch.save( nnet.state_dict(), os.path.join(out_base_dir,'epoch0_net.pth') )    # Save network at epoch 0 (i.e. at initialisation)
	nnet_file = open(os.path.join(out_base_dir,'epoch0_net.bin'),'wb')
	pk.dump(nnet,nnet_file,pk.HIGHEST_PROTOCOL)      
	nnet_file.close()


	###################### Run training ###################################

    
	start_time = time.time()

	loss_val_prev = np.inf
	for epoch in range(noepoch):
	    
		print('        EPOCH   {}/{}'.format(epoch+1,noepoch))
		print('')

		# Loop over mini-batches for at a fixed epoch
		minibatch_id = 0
		for signals in loadertrain:

			output = nnet(Tensor(signals))    # Pass MRI measurements through net and get estimates of MRI signals  
			lossmeas_train = nnetloss(output, signals)  # Training loss

			# Back propagation
			nnetopt.zero_grad()               # Evaluate loss gradient with respect to network parameters at the output layer
			lossmeas_train.backward()         # Backpropage the loss gradient through previous layers
			nnetopt.step()                    # Update network parameters
		
			losstrain[epoch,minibatch_id] = Tensor.numpy(lossmeas_train.data)
			minibatch_id = minibatch_id + 1
		
        

		nnet.eval()   # evaluation mode (deactivates dropout)
		dataval_nnet = nnet( Tensor(dataval) )                    # Output predicted MRI val signals

		tissueval_nnet = nnet.getparams( Tensor(dataval) )        # Estimated tissue val parameters
		neuronval_nnet = nnet.getneurons( Tensor(dataval) )       # Output neuron activations
        
		lossmeas_val = nnetloss( dataval_nnet , Tensor(dataval) ) # Validation loss

		dataval_nnet = dataval_nnet.detach().numpy()

		max_val_val_out = np.transpose( matlib.repmat(np.max(dataval_nnet,axis=1),nmeas_train,1) )

		lossval[epoch,0] = Tensor.numpy(lossmeas_val.data)


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
            
			# Update value of best validation loss
			loss_val_prev = Tensor.numpy(lossmeas_val.data)

			# Save predicted validation signals
			dataval_nnet = (max_val_val/max_val_val_out)*dataval_nnet   # Rescale signals
			dataval_nnet_file = open(os.path.join(out_base_dir,'lossvalmin_sigval.bin'),'wb')
			pk.dump(dataval_nnet,dataval_nnet_file,pk.HIGHEST_PROTOCOL)      
			dataval_nnet_file.close()
            
			# Save predicted validation tissue parameters 
			tissueval_nnet = tissueval_nnet.detach().numpy()
            
            
			tissueval_nnet[:,s0idx] = (max_val_val[:,0]/max_val_val_out[:,0])*tissueval_nnet[:,s0idx] # # Rescale s0 

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

				datatest_nnet = nnet( Tensor(datatest))              # Output of full network (predicted MRI signals)
				datatest_nnet = datatest_nnet.detach().numpy()
                
				max_val_test_out = np.transpose( matlib.repmat(np.max(datatest_nnet,axis=1),nmeas_train,1) )
				tissuetest_nnet = nnet.getparams( Tensor(datatest) )  # Estimated tissue parameters
				neurontest_nnet = nnet.getneurons( Tensor(datatest) ) # Output neuron activations
                
                
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
                
				file.write('Input measurements path: ' + str(args.data_train) + '\n' + '\n')
            
				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')
