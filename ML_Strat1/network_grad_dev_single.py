# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:43:19 2023

@author: pcastror
"""
import numpy as np
import math
import torch
from torch import nn
from torch import Tensor
#from torch import reshape as tshape
from torch import exp as texp 
from torch import log as tlog
from torch import abs as tabs
from numpy import matlib
from torch import reshape as tshape
from torch import cat as tcat
from torch import sin as tsin
from torch import cos as tcos
from torch import exp as texp
from torch import log as tlog
from torch import abs as tabs
from torch import erf as terf
from torch import sqrt as tsqrt
from torch import matmul as tmat
from torch import sum as tsum

 
### MRI net

# =============================================================================
# class mripar(nn.Module): 
# 	## MRI-Net: neural netork class to perform model fitting
# 
# 	# Constructor
# 	def __init__(self, nneurons, pdrop, mrimodel, mriprot, g):
# 		'''Initialise a Network to be trained on tissue parameters as:
# 			 
# 		   mynet = network.mripar (nneurons,pdrop,mrimodel,mriprot,g)
# 
# 		   * nneurons:     list or numpy array storing the number of neurons in each layer
#                         (e.g. [<INPUT NEURONS>,<HIDDEN NEURONS LAYER 1>, ... , <OUTPUT NEURONS>],
# 				              where the number of input neurons equals the number of MRI measurements
# 				              per voxel and the number of output neurons equal the nuber of tissue parameters
# 				              to estimate)
# 
# 		   * pdrop:        dropout probability for each layer of neurons in the net		 
#   
# 		   * mrimodel:     Zeppelin model. Zeppelin. Zeppelin without TE or T2 variations.
# 
# 				    			 
# 		   * mriprot:     MRI protocol stored as a numpy array. The protocol depends on the MRI model used. For
#                        Zeppelin model: mriprot is a matrix has size 2 x Nmeas, where Nmeas is the number
#                        of MRI measurements in the protocol, which must equal the number of input
#                        neurons (i.e. Nmeas must equal nneurons[0]). mriprot must contain:
# 					             -  mriprot[0,:]: b-values in s/mm^2
# 					             -  mriprot[1,:]: echo times TE in ms
#  
# 		'''
# 		super(mripar, self).__init__()
# 
# 		### Store input parameters
# 		layerlist = []   
# 		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers
# 		print('nlayers: ', nlayers)
# 
# 		self.mrimodel = mrimodel                    # MRI signal model. In this case Zeppelin
# 		self.mriprot = np.array(mriprot)            # MRI protocol. bvals
# 		self.nneurons = np.array(nneurons)          # Number of hidden neurons in each layer
# 		self.g = g                                  # gradient directions
# 		#self.grad_dev = grad_dev                    # gradient deviations
# 
# 
# 		# Model Zeppelin. Parameters: ang1, ang2, AD, RD, S0
# 		if self.mrimodel=='Zeppelin':
# 			if nneurons[-1]!=5:
# 				raise RuntimeError('the number of output neurons for model "Zeppelin" must be 5 and it was ', nneurons[-1])
# 			if mriprot.shape[0]!=2:
# 				raise RuntimeError('the size of the numpy matrix storting the MRI protocol is wrong')
# 		
# 			self.npars = 5
# 
# 		print('mriprot.shape[1] :', mriprot[0].shape[0])
# 		print('nneurons[0]: ', nneurons[0])
# 		print('nneurons: ', nneurons)
# 
# 		# General consistency of number of input neurons
# 		if  mriprot[0].shape[0]!=nneurons[0]:
# 			raise RuntimeError('the number of measurements in the MRI protocol must be equal to the number of input neurons')
# 					
# 		### Create layers 
# 
# 		#   Eeach elementary layer features linearity, ReLu and optional dropout
# 		for ll in range(nlayers):
# 			layerlist.append(nn.Linear(nneurons[ll], nneurons[ll+1]))   # Linearity. Creates a linear transformation layer (also known as a fully connected layer) 
# 
# 			layerlist.append(nn.ReLU(True))                             # Non-linearity
# 			if(ll<nlayers-1):
# 				layerlist.append(nn.Dropout(p=pdrop))                   # Dropout (no dropout for last layer)            
# 		
# 		layerlist.append(nn.Softplus())                                 # Add a softplus as last layer to make sure neuronal activations are never 0.0 (min. output: log(2.0)~0.6931)
# 
# 		# Store all layers
# 		print('layer list: ', layerlist)
# 		self.layers = nn.ModuleList(layerlist)
# 		print('layers: ', self.layers)
# 
# 
# 		# Add learnable normalisation factors to convert output neuron activations to tissue parameters 
# 		normlist = []
# 		for pp in range(self.npars):
# 			normlist.append( nn.Linear(1, 1, bias=False) )
# 		self.sgmnorm = nn.ModuleList(normlist)
# 
# 		#### Set upper and lower bounds for tissue parameters
# 
# 		# Model Zeppelin
# 		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0
# 
# 			S0_min =      0.0
# 			d_z_par_min = 0.01
# 			d_z_per_min = 0.0
# 			ang1_min =    0.0
# 			ang2_min =    0.0
# 
# 
# 			S0_max      = 5
# 			d_z_par_max = 3.2
# 			d_z_per_max = 1
# 			ang1_max    = np.pi
# 			ang2_max    = 2*np.pi-0.01
# 
# 
# 			self.param_min = np.array([ang1_min, ang2_min, d_z_par_min, d_z_per_min, S0_min]) 
# 			self.param_max = np.array([ang1_max, ang2_max, d_z_par_max, d_z_per_max, S0_max])
# 			self.param_name =  ['ang1', 'ang2', 'd_z_par', 'd_z_per', 'S0']
# 
# 
# 	### Calculator of output neuron activations from given MRI measurements
# 	def getneurons(self, x):
# 		''' Get output neuron activations from an initialised qMRI-Net
# 
# 		    u = mynet.getneurons(xin) 
# 
# 		    * mynet:  initialised qMRI-Net
# 
# 		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
# 			           (for a mini-batch, xin has size voxels x measurements)
# 
# 		    * u:      pytorch Tensor storing the output neuronal activations (if x is a mini-batch of size 
# 			           voxels x measurements, then u has size voxels x number_of_output_neurons; note that
# 			           there are as many output neurons as tissue parameters to estimate)  
#             
#             
#     		Pass MRI signals to layers and get output neuron activations. This for loop iterates over the layers of the neural network 
#             (self.layers) and sequentially passes the input x through each layer. The output of each layer becomes the input for the 
#             next layer, updating the value of x.
# 		'''	
# 
# 		for mylayer in self.layers:
# 			x = mylayer(x)
# 
# 		return x
# 
# 
# 	### Normalise output neurons to convert to tissue parameters
# 	def getnorm(self, x):
# 		''' Normalise output neuron activations before taking a sigmoid to convert to tissue parameters
# 
# 		    uout = mynet.getnorm(uin) 
# 
# 		    * mynet:  initialised qMRI-Net
# 
# 		    * uin:    pytorch Tensor storing output neuron activations for one voxels or for a mini-batch 
# 			           (for a mini-batch, xin has size voxels x measurements)
# 
# 		    * uout:   pytorch Tensor storing the normalised output neuronal activations (same size as uin)
# 		'''	
# 
# 		if x.dim()==1:  # Check if the dimensionality of x is 1, indicating that it contains the output neuron activations from a single voxel.
# 			
# 			# Construct a 1D tensor with one scaling factor per tissue parameter
# 			normt = Tensor( np.zeros( (self.npars) ) )
# 			for pp in range(self.npars):
# 				bt = np.zeros( (self.npars) )
# 				bt[pp] = 1.0
# 				bt = Tensor(bt)
# 				bt = self.sgmnorm[pp](Tensor([1.0]))*bt
# 				normt = normt + bt
# 
# 			# Normalise
# 			normt = tabs(normt)   # tabs function to ensure that all elements of normt are positive.
# 			x = x*normt
# 							
# 
# 		elif x.dim()==2:  # Check if the dimensionality of x is 2, indicating that it contains the output neuron activations from a mini-batch of multiple voxels.
# 
# 			# Construct a tensor with nvoxels x nparameters with scaling factors (all voxels have same scaling factors)
# 			normt = Tensor( np.zeros( (x.shape[0],self.npars) ) )
# 			for pp in range(self.npars):			
# 				bt = np.zeros( (x.shape[0],self.npars) )
# 				bt[:,pp] = 1.0
# 				bt = Tensor(bt)
# 				bt = self.sgmnorm[pp](Tensor([1.0]))*bt
# 				normt = normt + bt
# 
# 			# Normalise
# 			normt = tabs(normt)
# 			x = x*normt
# 
# 		else:
# 			raise RuntimeError('getnorm() processes 1D (signals from one voxels) or 2D (signals from multiple voxels) inputs!') 
# 
# 
# 		return x
# 
# 
# 
# 	### Estimator of tissue parameters from given MRI measurements
# 	def getparams(self, x):
# 		''' Get tissue parameters from an initialised qMRI-Net
# 
# 		    p = mynet.getparams(uin) 
# 
# 		    * mynet:  initialised qMRI-Net
# 
# 		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
# 			           (for a mini-batch, xin has size voxels x measurements)
# 
# 		    * p:      pytorch Tensor storing the predicted tissue parameters (if x is a mini-batch of size 
# 			           voxels x measurements, then p has size voxels x number_of_tissue_parameters)  
# 		'''	
# 
# 		## Pass MRI signals to layers and get output neuronal activations
# 		x = self.getneurons(x)   # Note that the last layer is softplus, so x cannot be 0.0
# 
# 		## Normalise neuronal activations in [0; 1.0] in 4 steps (A, B, C, D):
# 		x = tlog(x)                                      # step A: take log as activations can vary over many orders of magnitude
# 		x = x - tlog(tlog(Tensor([2.0])))                # step B: as the minimum value for x was log(2.0), make sure x is not negative
# 		x = self.getnorm(x)                              # step C: multiply each parameter for a learnable normalisation factor
# 		x = 2.0*( 1.0 / (1.0 + texp(-x) ) - 0.5 );       # step D: pass through a sigmoid
# 
# 		## Map normalised neuronal activations to MRI tissue parameter ranges
# 		
# 		# Model Zeppelin_		
# 		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0
# 
# 			# Single voxels
# 			if x.dim()==1:
# 				for pp in range(0,self.npars):
# 					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]
# 
# 			# Mini-batch from multiple voxels
# 			elif x.dim()==2:	
# 
# 				t_allones = np.ones(( x.shape[0], 1)) 
# 
# 				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones , self.param_max[4]*t_allones) , axis=1 )  )
# 				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones , self.param_min[4]*t_allones) , axis=1 )  )
# 
# 
# 				x = (max_val - min_val)*x + min_val
# 
# 
# 		# Return tissue parameters
# 		return x
# 
# 
# 	### Computation of MRI signals from tissue parameters
# 	def getsignals(self, x):
# 		''' Get MRI signals from tissue parameters using analytical MRI signal models
# 
# 		    xout = mynet.getsignals(pin) 
# 
# 		    * mynet:  initialised qMRI-Net
# 
# 		    * pin:    pytorch Tensor storing the tissue parameters from one voxels or from a mini-batch
# 			           (if pin is a mini-batch of multiple voxels, it must have size 
# 			           voxels x number_of_tissue_parameters)
# 
# 		    * xout:   pytorch Tensor storing the predicted MRI signals given input tissue parameters and
# 			           according to the analytical model specificed in field mynet.mrimodel  
# 			           (for a mini-batch of multiple voxels, xout has size voxels x measurements)  
# 		'''		
# 
# 		## Get MRI protocol
# 		mriseq = np.copy(self.mriprot)   # Sequence parameters
# 
# 		## Compute MRI signals from input parameter x (microstructural tissue parameters) 
# 		# Model Zeppelin
# 		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
# 			
# 			bval = Tensor(mriseq[0])
# 			g =  Tensor(self.g)
# 			x_tot = torch.empty(x.shape[0], g.size(0))
#             
# 			for ii in range(0,x.shape[0],1):        
# 				b_delta = 1
# 				bval = bval.view(1, -1).t()
# 				term0 = torch.stack([tcos(x[ii,1])*tsin(x[ii,0]), tsin(x[ii,0])*tsin(x[ii,1]), tcos(x[ii,0])])
# 				term1 = Tensor.tile(term0, [g.size(0), 1])
# 				dot_prod = (g * term1).sum(axis=1, keepdims=True)
# 				S = x[ii,4] * texp(b_delta * (1/3) * bval * (x[ii,2]-x[ii,3]*x[ii,2]) - bval * (1/3) * (x[ii,3]*x[ii,2]+2*x[ii,2]) - bval * b_delta * dot_prod**2 * (x[ii,2]-x[ii,3]*x[ii,2]))
# 
# 				S = S.flatten()
# 				x_tot[ii,:] = S 
# 
# 		x = x_tot
# 
# 		# Return predicted MRI signals
# 		return x
# 
# 
# 	### Change limits of tissue parameters    
# 	def changelim(self, pmin, pmax):
# 		'''Change limits of tissue parameters
# 		
# 		    mynet.changelim(pmin, pmax) 
# 		
# 		    * mynet:  initialised MRI-Net
# 		 
# 		    * pmin:   array of new lower bounds for tissue parameters
# 		              (5 parameters if model is Zeppelin)
# 		               
# 		    * pmax:   array of new upper bounds for tissue parameters
# 		              (5 parameters if model is Zeppelin)
# 		'''	
# 
# 		# Check number of tissue parameters
# 		if (len(pmin) != self.npars) or (len(pmax) != self.npars):
# 			raise RuntimeError('you need to specify bounds for {} tissue parameters with model {}'.format(self.npars,self.mrimodel))
# 
# 
# 		# Model Zeppelin
# 		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
#             
# 			ang1_min =    [0]
# 			ang2_min =    [1]
# 			d_z_par_min = [2]
# 			d_z_per_min = [3]
# 			S0_min =      [4]
# 
# 			ang1_max =    [0]
# 			ang2_max =    [1]
# 			d_z_par_max = [2]
# 			d_z_per_max = [3]
# 			S0_max =      [4]
# 
# 
# 			self.param_min = np.array([ang1_min, ang2_min, d_z_par_min, d_z_per_min, S0_min]) 
# 			self.param_max = np.array([ang1_max, ang2_max, d_z_par_max, d_z_per_max, S0_max])
#          
#   
# 
# 	### Predictor of MRI signal given a set of MRI measurements    
# 	def forward(self, x):
# 		''' Full forward pass of a qMRI-Net, estimating MRI signals given input MRI measurements
# 
# 		    y = mynet.forward(x) 
# 		    y = mynet(x) 
# 
# 		    * mynet:  initialised qMRI-Net
# 
# 		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
# 			           (for a mini-batch of multiple voxels, x has size voxels x measurements)
# 
# 		    * y:      pytorch Tensor storing a prediction of x (it has same size as x)
# 		'''	
# 
# 		## Get prediction of tissue parameters from MRI measurements
# 		x = self.getparams(x)    
# 
# 		## Return tissue parameters
# 		return x
# =============================================================================
    
    
class mrisig(nn.Module):

	# Constructor
	def __init__(self,nneurons,pdrop,mrimodel,bvals, bvecs): 
        
        # constructor of the qmrisig class. It is called when a new instance of the class is created. 
        # The constructor initializes the qMRI-Net with the specified parameters. 
        
		'''Initialise a MRI-Net to be trained on MRI signals as:
			 
		   mynet = deepqmri.qmrisig(nneurons,pdrop,mrimodel,mriprot)


		   * nneurons:     list or numpy array storing the number of neurons in each layer
                        (e.g. [<INPUT NEURONS>,<HIDDEN NEURONS LAYER 1>, ... , <OUTPUT NEURONS>],
				              where the number of input neurons equals the number of MRI measurements
				              per voxel and the number of output neurons equal the nuber of tissue parameters
				              to estimate)

		   * pdrop:        dropout probability for each layer of neurons in the net. Dropout is a regularization technique used to 
                        prevent overfitting by randomly setting a fraction of inputs to zero during training.	

		   * mrimodel:     a string identifying the MRI model to fit.
				    			    						

		   * mriprot:     MRI protocol stored as a numpy array. The protocol depends on the MRI model used. For
                       Zeppelin model: mriprot is a matrix has size 2 x Nmeas, where Nmeas is the number
                       of MRI measurements in the protocol, which must equal the number of input
                       neurons (i.e. Nmeas must equal nneurons[0]). mriprot must contain:
					             -  mriprot[0,:]: b-values in s/mm^2
					             -  mriprot[1,:]: echo times TE in ms



		'''
		super(mrisig, self).__init__()    # call the constructor of the parent class nn.Module to properly initialize the qmrisig class.
        ## Inheritance from the super class nn from pytorch
		
		### Store input parameters
		layerlist = []   
		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers

		print('nlayers: ', nlayers)
		self.mrimodel = mrimodel                    # MRI signal model
		self.bvals = bvals                      # MRI protocol
		self.bvecs = bvecs                                  # gradient directions
		#self.grad_dev = grad_dev                    # gradient deviations

		self.nneurons = np.array(nneurons)          # Number of hidden neurons in each layer
 

		# Model Zeppelin
		if self.mrimodel=='Zeppelin':
			if nneurons[-1]!=5:
				raise RuntimeError('the number of output neurons for model "Zeppelin" must be 5 and it was ', nneurons[-1])

			self.npars = 5		

		# General consistency of number of input neurons
		if  bvals.shape!=nneurons[0]:
			raise RuntimeError('the number of measurements in the MRI protocol must be equal to the number of input neurons')
					

		### Create layers 
        
		for ll in range(nlayers):
			layerlist.append(nn.Linear(nneurons[ll], nneurons[ll+1]))   # Linearity
			layerlist.append(nn.ReLU(True))                             # Non-linearity. ReLU applies the element-wise activation function max(0, x) to the output of the linear layer, introducing non-linearity.
			if(ll<nlayers-1):
				layerlist.append(nn.Dropout(p=pdrop))                   # Dropout (no dropout for last layer). Dropout randomly sets a fraction of inputs to zero during training, which helps prevent overfitting.

		layerlist.append(nn.Softplus())                                 # Add a softplus as last layer to make sure neuronal activations are never 0.0 (min. output: log(2.0)~0.6931

		# Store layers
		self.layers = nn.ModuleList(layerlist)

		# Add learnable normalisation factors to convert output neuron activations to tissue parameters 
		normlist = []
		for pp in range(self.npars):
			normlist.append( nn.Linear(1, 1, bias=False) )
		self.sgmnorm = nn.ModuleList(normlist)


		#### Set upper and lower bounds for tissue parameters depending on the signal model
        
		# Model Zeppelin
		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0

			S0_min =      0.0
			d_z_par_min = 0.01
			d_z_per_min = 0.0
			ang1_min =    0.0
			ang2_min =    0.0


			S0_max      = 5
			d_z_par_max = 3.2
			d_z_per_max = 1
			ang1_max    = np.pi
			ang2_max    = 2*np.pi-0.01


			self.param_min = np.array([ang1_min, ang2_min, d_z_par_min, d_z_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_z_par_max, d_z_per_max, S0_max])
			self.param_name =  ['ang1', 'ang2', 'd_z_par', 'd_z_per', 'S0']



	### Calculator of output neuron activations from given MRI measurements
	def getneurons(self, x):
		''' Get output neuron activations from an initialised qMRI-Net

		    u = mynet.getneurons(xin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * u:      pytorch Tensor storing the output neuronal activations (if x is a mini-batch of size 
			           voxels x measurements, then u has size voxels x number_of_output_neurons; note that
			           there are as many output neurons as tissue parameters to estimate) 
            
    		Pass MRI signals to layers and get output neuron activations. This for loop iterates over the layers of the neural network 
            (self.layers) and sequentially passes the input x through each layer. The output of each layer becomes the input for the 
            next layer, updating the value of x.
		'''	

		for mylayer in self.layers:
			x = mylayer(x)

		## Return neuron activations
		return x


	### Normalise output neurons to convert to tissue parameters
	def getnorm(self, x):
		''' Normalise output neuron activations before taking a sigmoid to convert to tissue parameters

		    uout = mynet.getnorm(uin) 

		    * mynet:  initialised qMRI-Net

		    * uin:    pytorch Tensor storing output neuron activations for one voxels or for a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * uout:   pytorch Tensor storing the normalised output neuronal activations (same size as uin)
		'''	

		if x.dim()==1:  # Check if the dimensionality of x is 1, indicating that it contains the output neuron activations from a single voxel.
			
			# Construct a 1D tensor with one scaling factor per tissue parameter
			normt = Tensor( np.zeros( (self.npars) ) )
			for pp in range(self.npars):
				bt = np.zeros( (self.npars) )
				bt[pp] = 1.0
				bt = Tensor(bt)
				bt = self.sgmnorm[pp](Tensor([1.0]))*bt
				normt = normt + bt

			# Normalise
			normt = tabs(normt)   # tabs function to ensure that all elements of normt are positive.
			x = x*normt
							


		elif x.dim()==2:  # Check if the dimensionality of x is 2, indicating that it contains the output neuron activations from a mini-batch of multiple voxels.

			# Construct a tensor with nvoxels x nparameters with scaling factors (all voxels have same scaling factors)
			normt = Tensor( np.zeros( (x.shape[0],self.npars) ) )
			for pp in range(self.npars):			
				bt = np.zeros( (x.shape[0],self.npars) )
				bt[:,pp] = 1.0
				bt = Tensor(bt)
				bt = self.sgmnorm[pp](Tensor([1.0]))*bt
				normt = normt + bt

			# Normalise
			normt = tabs(normt)
			x = x*normt

		else:
			raise RuntimeError('getnorm() processes 1D (signals from one voxels) or 2D (signals from multiple voxels) inputs!') 


		# Return scaled neuron activation
		return x



	### Estimator of tissue parameters from given MRI measurements
	def getparams(self, x):
		''' Get tissue parameters from an initialised qMRI-Net

		    p = mynet.getparams(uin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * p:      pytorch Tensor storing the predicted tissue parameters (if x is a mini-batch of size 
			           voxels x measurements, then p has size voxels x number_of_tissue_parameters)  
		'''	
		#print('shape input measurements: ', x.shape)

		x_measurements = x[:,0:(self.bvecs).shape[0]]
		#print('x_measurements: ', x_measurements)
		x_grad_dev = x[:,(self.bvecs).shape[0]:]
		#print('x_grad_dev: ', x_grad_dev)

		## Pass MRI signals to layers and get output neuronal activations
		x = self.getneurons(x_measurements)   # Note that the last layer is softplus, so x cannot be 0.0

		## Normalise neuronal activations in [0; 1.0] in 4 steps (A, B, C, D):
		x = tlog(x)                                      # step A: take log as activations can vary over many orders of magnitude
		x = x - tlog(tlog(Tensor([2.0])))                # step B: as the minimum value for x was log(2.0), make sure x is not negative
		x = self.getnorm(x)                              # step C: multiply each parameter for a learnable normalisation factor
		x = 2.0*( 1.0 / (1.0 + texp(-x) ) - 0.5 );       # step D: pass through a sigmoid

		## Map normalised neuronal activations to MRI tissue parameter ranges
		# Model Zeppelin		
		if self.mrimodel=='Zeppelin': # Parameter order:  ang1, ang2, AD, RD, S0

			# Single voxels
			if x.dim()==1:
				for pp in range(0,self.npars):
					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]

			# Mini-batch from multiple voxels
			elif x.dim()==2:	

				t_allones = np.ones(( x.shape[0], 1)) 

				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones , self.param_max[4]*t_allones) , axis=1 )  )
				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones , self.param_min[4]*t_allones) , axis=1 )  )


				x = (max_val - min_val)*x + min_val

		#print('shape parameters: ', x.shape)
		# Return tissue parameters
		return x, x_grad_dev



	### Computation of MRI signals from tissue parameters
	def getsignals(self,x, x_grad_dev):
		''' Get MRI signals from tissue parameters using analytical MRI signal models

		    xout = mynet.getsignals(pin) 

		    * mynet:  initialised qMRI-Net

		    * pin:    pytorch Tensor storing the tissue parameters from one voxels or from a mini-batch
			           (if pin is a mini-batch of multiple voxels, it must have size 
			           voxels x number_of_tissue_parameters)

		    * xout:   pytorch Tensor storing the predicted MRI signals given input tissue parameters and
			           according to the analytical model specificed in field mynet.mrimodel  
			           (for a mini-batch of multiple voxels, xout has size voxels x measurements)  
		'''		

		## Get MRI protocol
		#print('x dim(): ', x.dim())
		## Compute MRI signals from input parameter x (microstructural tissue parameters) 
		# Model Zeppelin
		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
			
			bval = (self.bvals)
			g =  (self.bvecs)
			x_tot = torch.empty(x.shape[0], g.shape[0])
            
			for ii in range(0,x.shape[0],1): 
                
                # construct L matrix
				L = np.array([[x_grad_dev[ii,0], x_grad_dev[ii,1], x_grad_dev[ii,2]],
                                  [x_grad_dev[ii,3], x_grad_dev[ii,4], x_grad_dev[ii,5]],
                                  [x_grad_dev[ii,6], x_grad_dev[ii,7], x_grad_dev[ii,8]]])
                 
                # Apply gradient non-linearities correction
				I = np.eye(3)                          # identity matrix               
				v = np.dot(g, (I + L))
				n = np.sqrt(np.diag(np.dot(v, v.T)))
                
				new_bvec = v / np.tile(n, (3, 1)).T    # normalize bvecs
				new_bval = n ** 2 * bval
				new_bvec[new_bval == 0, :] = 0
				bval_corrected = Tensor(new_bval)
				g_effective = Tensor(new_bvec)
                

				b_delta = 1
				bval_corrected = bval_corrected.view(1, -1).t()
				term0 = torch.stack([tcos(x[ii,1])*tsin(x[ii,0]), tsin(x[ii,0])*tsin(x[ii,1]), tcos(x[ii,0])])
				term1 = Tensor.tile(term0, [g_effective.size(0), 1])
				dot_prod = (g_effective * term1).sum(axis=1, keepdims=True)
				S = x[ii,4] * texp(b_delta * (1/3) * bval_corrected * (x[ii,2]-x[ii,3]*x[ii,2]) - bval_corrected * (1/3) * (x[ii,3]*x[ii,2]+2*x[ii,2]) - bval_corrected * b_delta * dot_prod**2 * (x[ii,2]-x[ii,3]*x[ii,2]))

				S = S.flatten()
				x_tot[ii,:] = S 

		x = x_tot
        
		return x

	### Change limits of tissue parameters    
	def changelim(self, pmin, pmax):
		''' Change limits of tissue parameters
		
		    mynet.changelim(pmin, pmax) 
		
		    * mynet:  initialised qMRI-Net
		 
		    * pmin:   array of new lower bounds for tissue parameters
		              (5 parameters if model is Zeppelin)
		               
		    * pmax:   array of new upper bounds for tissue parameters
		              (5 parameters if model is Zeppelin)
		'''	

		# Check number of tissue parameters
		if (len(pmin) != self.npars) or (len(pmax) != self.npars):
			raise RuntimeError('you need to specify bounds for {} tissue parameters with model {}'.format(self.npars,self.mrimodel))


		# Model Zeppelin
		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
            
			ang1_min =    [0]
			ang2_min =    [1]
			d_z_par_min = [2]
			d_z_per_min = [3]
			S0_min =      [4]

			ang1_max =    [0]
			ang2_max =    [1]
			d_z_par_max = [2]
			d_z_per_max = [3]
			S0_max =      [4]


			self.param_min = np.array([ang1_min, ang2_min, d_z_par_min, d_z_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_z_par_max, d_z_per_max, S0_max])
	

	### Predictor of MRI signal given a set of MRI measurements    
	def forward(self, x):
		''' Full forward pass of a qMRI-Net, estimating MRI signals given input MRI measurements

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  initialised qMRI-Net

		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
			      (for a mini-batch of multiple voxels, x has size voxels x measurements)

		    * y:      pytorch Tensor storing a prediction of x (it has same size as x)
		'''	

		## Encoder: get prediction of underlying tissue parameters
		encoded2, x_grad_dev = self.getparams(x)    # From MRI measurements --> to tissue parameters
		#print('tissue param: ', encoded2)
		## Decoder: predict MRI signal back from the estimated tissue parameters
		decoded = self.getsignals(encoded2, x_grad_dev)   # From tissue parameters --> to predicted MRI signals 
		#print('MRI signals: ', x.shape)

		## Return predicted MRI signals given the set of input measurements
		return decoded

    




