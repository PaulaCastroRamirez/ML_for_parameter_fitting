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

 
### MRI network for single bmatrix. Model defined with the bmatrix!

class mripar(nn.Module): 

	def __init__(self, nneurons, pdrop, mrimodel, bvals, bvecs):
		'''Initialise a Network to be trained on tissue parameters as:
			 
		   mynet = network.mripar (nneurons, pdrop, mrimodel, bvals, bvecs)

		   * nneurons:     list or numpy array storing the number of neurons in each layer
		   * pdrop:        dropout probability for each layer of neurons in the net		 
		   * mrimodel:     Zeppelin without TE or T2 variations.
		   * bvals:        bvalues
		   * bvecs:        gradient directions
 
		'''
		super(mripar, self).__init__()
	
		### Store input parameters
		layerlist = []   
		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers
		print('nlayers: ', nlayers)

		self.mrimodel = mrimodel                    # MRI signal model. In this case Zeppelin
		self.bvals = np.array(bvals)                # MRI protocol. bvals
		self.nneurons = np.array(nneurons)          # Number of hidden neurons in each layer
		self.bvecs = bvecs                          # gradient directions


		# Model Zeppelin. Parameters: ang1, ang2, AD, RD, S0
		if self.mrimodel=='Zeppelin':
			if nneurons[-1]!=5:
				raise RuntimeError('the number of output neurons for model "Zeppelin" must be 5 and it was ', nneurons[-1])
	
			self.npars = 5

		print('nneurons in the model: ', nneurons)

		# General consistency of number of input neurons
		if  bvals.shape[0]!=nneurons[0]:
			raise RuntimeError('the number of measurements in the MRI protocol must be equal to the number of input neurons')
					
		### Create layers 
		for ll in range(nlayers):
			layerlist.append(nn.Linear(nneurons[ll], nneurons[ll+1]))   # Linearity. Creates a linear transformation layer (also known as a fully connected layer) 

			layerlist.append(nn.ReLU(True))                             # Non-linearity
			if(ll<nlayers-1):
				layerlist.append(nn.Dropout(p=pdrop))                   # Dropout (no dropout for last layer)            
		
		layerlist.append(nn.Softplus())                                 # Add a softplus as last layer to make sure neuronal activations are never 0.0 (min. output: log(2.0)~0.6931)

		# Store all layers
		print('layer list: ', layerlist)
		self.layers = nn.ModuleList(layerlist)

		# Add learnable normalisation factors to convert output neuron activations to tissue parameters 
		normlist = []
		for pp in range(self.npars):
			normlist.append( nn.Linear(1, 1, bias=False) )
		self.sgmnorm = nn.ModuleList(normlist)

		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0

			S0_min    = 0.0
			d_par_min = 0.01
			k_per_min = 0.0
			ang1_min  = 0.0
			ang2_min  = 0.0


			S0_max    = 5
			d_par_max = 3.2
			k_per_max = 1
			ang1_max  = np.pi
			ang2_max  = 2*np.pi-0.01


			self.param_min = np.array([ang1_min, ang2_min, d_par_min, k_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_par_max, k_per_max, S0_max])
			self.param_name =  ['ang1', 'ang2', 'd_par', 'k_per', 'S0']


	def getneurons(self, x):
		''' Get output neuron activations from an initialised qMRI-Net

		    u = mynet.getneurons(xin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * u:      pytorch Tensor storing the output neuronal activations (if x is a mini-batch of size 
			           voxels x measurements, then u has size voxels x number_of_output_neurons)
		'''	

		for mylayer in self.layers:
			x = mylayer(x)

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
				bt = self.sgmnorm[pp](Tensor([1.0])) *bt
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
		x = self.getneurons(x)   

		## Normalise neuronal activations in [0 1] 
		x = tlog(x)                                      
		x = x - tlog(tlog(Tensor([2.0])))               
		x = self.getnorm(x)                              # multiply each parameter for a learnable normalisation factor
		x = 2.0*( 1.0 / (1.0 + texp(-x) ) - 0.5 );       # modified tanh

		# Model Zeppelin. Map normalised neuronal activations to MRI tissue parameter ranges
		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0
			if x.dim()==1:
				for pp in range(0,self.npars):
					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]

			elif x.dim()==2:	

				t_allones = np.ones(( x.shape[0], 1)) 

				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones , self.param_max[4]*t_allones) , axis=1 )  )
				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones , self.param_min[4]*t_allones) , axis=1 )  )

				x = (max_val - min_val)*x + min_val


		return x


	### Computation of MRI signals from tissue parameters
	def getsignals(self, x):
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

		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
        
			b_delta = 1
			
			# Repeat bvalues Nmeas times
			bval = Tensor(np.tile(self.bvals, (x.shape[0], 1)))                   # Shape: (N, 108)
			# Repeat bvec Nmeas times
			g = Tensor(np.tile(self.bvecs[np.newaxis, ...], (x.shape[0], 1, 1)))  # Shape: (N, 108, 3)
                      
			if x.dim()==1:
                
				Nmeas = bval.shape[0]
				Nvox = 1

				x1 = tshape(g[:,0], (1,Nmeas))
				x2 = tshape(g[:,1], (1,Nmeas))
				x3 = tshape(g[:,2], (1,Nmeas))

				angles_dprod = tmat(tshape(torch.sin(x[0])*torch.cos(x[1]), (Nvox,1)), x1) + tmat(tshape(torch.sin(x[1])*torch.sin(x[0]), (Nvox,1)), x2) + tmat(tshape(torch.cos(x[0]), (Nvox,1)), x3)
				b_D = b_delta / 3.0 * bval * (x[2] - x[3]*x[2]) - bval / 3.0 * (x[3]*x[2] + 2.0 * x[2]) - bval * b_delta * (torch.square(angles_dprod) * (x[2] - x[3]*x[2]))

				s_tot = x[4] * texp(b_D)
    
				x = 1.0*s_tot
                
           		
			elif x.dim()==2:

				Nmeas = bval.shape[1]
				Nvox = x.shape[0]
                
				x1 = tshape(g[:,:,0], (Nvox,Nmeas))
				x2 = tshape(g[:,:,1], (Nvox,Nmeas))
				x3 = tshape(g[:,:,2], (Nvox,Nmeas))
                
				bval = tshape(bval, (Nvox,Nmeas))
                
				b_delta = 1
				ones_ten = torch.ones(1,Nmeas, dtype=torch.float32)

    
				b_D = b_delta / 3.0 * (tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)) * bval)
				b_D = b_D - 1.0 / 3.0 * (tshape(x[:,3]*x[:,2] + 2.0 * x[:,2], (Nvox,1)) * bval)
				angles_dprod = (tshape(torch.sin(x[:,0])*torch.cos(x[:,1]), (Nvox,1)) * x1) + (tshape(torch.sin(x[:,1])*torch.sin(x[:,0]), (Nvox,1)) * x2) + (tshape(torch.cos(x[:,0]), (Nvox,1)) * x3)
				b_D_term3 = (tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)) * bval) * torch.square(angles_dprod)
				b_D = b_D - b_delta * b_D_term3

				s_tot = (tshape(x[:,4],(Nvox,1)) * ones_ten)
    
				s_tot = s_tot * texp(b_D)

				x = 1.0*s_tot

		return x


	### Change limits of tissue parameters    
	def changelim(self, pmin, pmax):
		'''Change limits of tissue parameters
		
		    mynet.changelim(pmin, pmax) 
		
		    * mynet:  initialised MRI-Net
		 
		    * pmin:   array of new lower bounds for tissue parameters
		              (5 parameters if model is Zeppelin)
		               
		    * pmax:   array of new upper bounds for tissue parameters
		              (5 parameters if model is Zeppelin)
		'''	

		if (len(pmin) != self.npars) or (len(pmax) != self.npars):
			raise RuntimeError('you need to specify bounds for {} tissue parameters with model {}'.format(self.npars,self.mrimodel))


		# Model Zeppelin
		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
            
			ang1_min  = [0]
			ang2_min  = [1]
			d_par_min = [2]
			k_per_min = [3]
			S0_min    = [4]

			ang1_max  = [0]
			ang2_max  = [1]
			d_par_max = [2]
			k_per_max = [3]
			S0_max    = [4]


			self.param_min = np.array([ang1_min, ang2_min, d_par_min, k_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_par_max, k_per_max, S0_max])
         
  

	def forward(self, x):
		''' Full forward pass of a MRI-Net, estimating MRI signals given input MRI measurements

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  initialised qMRI-Net

		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
			           (for a mini-batch of multiple voxels, x has size voxels x measurements)

		    * y:      pytorch Tensor storing a prediction of x (it has same size as x)
		'''	

		x = self.getparams(x)    

		return x
    
    
class mrisig(nn.Module):

	def __init__(self, nneurons, pdrop, mrimodel, bvals, bvecs): 
        
		'''Initialise a Network to be trained on tissue parameters as:
			 
		   mynet = network.mripar (nneurons, pdrop, mrimodel, bvals, bvecs)

		   * nneurons:     list or numpy array storing the number of neurons in each layer
		   * pdrop:        dropout probability for each layer of neurons in the net		 
		   * mrimodel:     Zeppelin without TE or T2 variations.
		   * bvals:        bvalues
		   * bvecs:        gradient directions

		'''
		super(mrisig, self).__init__()    # call the constructor of the parent class nn.Module to properly initialize the qmrisig class.
        ## Inheritance from the super class nn from pytorch
		
		### Store input parameters
		layerlist = []   
		nlayers = np.array(nneurons).shape[0] - 1   # Number of layers

		print('nlayers: ', nlayers)
		self.mrimodel = mrimodel                    # MRI signal model
		self.bvals = bvals                          # MRI protocol. bvals
		self.bvecs = bvecs                          # gradient directions

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
			if ll == nlayers - 1:  
				layerlist.append(nn.Softplus())                         # Add a softplus as last layer to make sure neuronal activations are never 0.0 (min. output: log(2.0)~0.6931

		self.layers = nn.ModuleList(layerlist)

		# Add learnable normalisation factors to convert output neuron activations to tissue parameters 
		normlist = []
		for pp in range(self.npars):
			normlist.append( nn.Linear(1, 1, bias=False) )
		self.sgmnorm = nn.ModuleList(normlist)
        
		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0

			S0_min    = 0.0
			d_par_min = 0.01
			k_per_min = 0.0
			ang1_min  = 0.0
			ang2_min  = 0.0


			S0_max    = 5
			d_par_max = 3.2
			k_per_max = 1
			ang1_max  = np.pi
			ang2_max  = 2*np.pi-0.01


			self.param_min = np.array([ang1_min, ang2_min, d_par_min, k_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_par_max, k_per_max, S0_max])
			self.param_name =  ['ang1', 'ang2', 'd_z_par', 'd_z_per', 'S0']


	def getneurons(self, x):
		''' Get output neuron activations from an initialised qMRI-Net

		    u = mynet.getneurons(xin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * u:      pytorch Tensor storing the output neuronal activations (if x is a mini-batch of size 
			           voxels x measurements, then u has size voxels x number_of_output_neurons)
		'''	

		for mylayer in self.layers:
			x = mylayer(x)

		return x


	def getnorm(self, x):
		''' Normalise output neuron activations before taking a sigmoid to convert to tissue parameters

		    uout = mynet.getnorm(uin) 

		    * mynet:  initialised qMRI-Net

		    * uin:    pytorch Tensor storing output neuron activations for one voxels or for a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * uout:   pytorch Tensor storing the normalised output neuronal activations (same size as uin)
		'''	

		if x.dim()==1:  
			
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
							


		elif x.dim()==2:  

			# Construct a tensor with nvoxels x nparameters with scaling factors (all voxels have same scaling factors)
			normt = Tensor( np.zeros( (x.shape[0],self.npars)))
			for pp in range(self.npars):			
				bt = np.zeros( (x.shape[0],self.npars) )
				bt[:,pp] = 1.0
				bt = Tensor(bt)
				bt = self.sgmnorm[pp](Tensor([1.0])) * bt
				normt = normt + bt

			# Normalise
			normt = tabs(normt)
			x = x*normt

		else:
			raise RuntimeError('getnorm() processes 1D (signals from one voxels) or 2D (signals from multiple voxels) inputs!') 

		return x


	def getparams(self, x):
		''' Get tissue parameters from an initialised qMRI-Net

		    p = mynet.getparams(uin) 

		    * mynet:  initialised qMRI-Net

		    * xin:    pytorch Tensor storing MRI measurements from one voxels or from a mini-batch 
			           (for a mini-batch, xin has size voxels x measurements)

		    * p:      pytorch Tensor storing the predicted tissue parameters (if x is a mini-batch of size 
			           voxels x measurements, then p has size voxels x number_of_tissue_parameters)  
		'''	

		x = self.getneurons(x)   

		## Normalise neuronal activations in [0 1] 
		x = tlog(x)                                    
		x = x - tlog(tlog(Tensor([2.0])))              
		x = self.getnorm(x)                              # multiply each parameter for a learnable normalisation factor
		x = 2.0*( 1.0 / (1.0 + texp(-x) ) - 0.5 );       # pass through a modified tanh

		# Model Zeppelin		
		if self.mrimodel=='Zeppelin': # Parameter order:  ang1, ang2, AD, RD, S0

			if x.dim()==1:  # Single voxels
				for pp in range(0,self.npars):
					x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]

			elif x.dim()==2:   # Mini-batch from multiple voxels

				t_allones = np.ones(( x.shape[0], 1)) 

				max_val = Tensor( np.concatenate( ( self.param_max[0]*t_allones , self.param_max[1]*t_allones , self.param_max[2]*t_allones , self.param_max[3]*t_allones , self.param_max[4]*t_allones) , axis=1 )  )
				min_val = Tensor( np.concatenate( ( self.param_min[0]*t_allones , self.param_min[1]*t_allones , self.param_min[2]*t_allones , self.param_min[3]*t_allones , self.param_min[4]*t_allones) , axis=1 )  )

				x = (max_val - min_val)*x + min_val


		return x


	def getsignals(self,x):
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
		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
        
			b_delta = 1
			
			# Repeat bvalues Nmeas times
			bval = Tensor(np.tile(self.bvals, (x.shape[0], 1)))                  # Shape: (N, 108)
			# Repeat bvec Nmeas times
			g = Tensor(np.tile(self.g[np.newaxis, ...], (x.shape[0], 1, 1)))  # Shape: (N, 108, 3)
			
			
			if x.dim()==1:
                
				Nmeas = bval.shape[0]
				Nvox = 1

				x1 = tshape(g[:,0], (1,Nmeas))
				x2 = tshape(g[:,1], (1,Nmeas))
				x3 = tshape(g[:,2], (1,Nmeas))

				angles_dprod = tmat(tshape(torch.sin(x[0])*torch.cos(x[1]), (Nvox,1)), x1) + tmat(tshape(torch.sin(x[1])*torch.sin(x[0]), (Nvox,1)), x2) + tmat(tshape(torch.cos(x[0]), (Nvox,1)), x3)
				b_D = b_delta / 3.0 * bval * (x[2] - x[3]*x[2]) - bval / 3.0 * (x[3]*x[2] + 2.0 * x[2]) - bval * b_delta * (torch.square(angles_dprod) * (x[2] - x[3]*x[2]))

				s_tot = x[4] * texp(b_D)
    
				x = 1.0*s_tot
                
           		
			elif x.dim()==2:

				Nmeas = bval.shape[1]
				Nvox = x.shape[0]
                
				x1 = tshape(g[:,:,0], (Nvox,Nmeas))
				x2 = tshape(g[:,:,1], (Nvox,Nmeas))
				x3 = tshape(g[:,:,2], (Nvox,Nmeas))
                
				bval = tshape(bval, (Nvox,Nmeas))
                
				b_delta = 1
				ones_ten = torch.ones(1,Nmeas, dtype=torch.float32)

    
				b_D = b_delta / 3.0 * (tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)) * bval)
				b_D = b_D - 1.0 / 3.0 * (tshape(x[:,3]*x[:,2] + 2.0 * x[:,2], (Nvox,1)) * bval)
				angles_dprod = (tshape(torch.sin(x[:,0])*torch.cos(x[:,1]), (Nvox,1)) * x1) + (tshape(torch.sin(x[:,1])*torch.sin(x[:,0]), (Nvox,1)) * x2) + (tshape(torch.cos(x[:,0]), (Nvox,1)) * x3)
				b_D_term3 = (tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)) * bval) * torch.square(angles_dprod)
				b_D = b_D - b_delta * b_D_term3

				s_tot = (tshape(x[:,4],(Nvox,1)) * ones_ten)
    
				s_tot = s_tot * texp(b_D)

				x = 1.0*s_tot

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

		if (len(pmin) != self.npars) or (len(pmax) != self.npars):
			raise RuntimeError('you need to specify bounds for {} tissue parameters with model {}'.format(self.npars,self.mrimodel))
			
		# Model Zeppelin
		if self.mrimodel=='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
            
			ang1_min  = [0]
			ang2_min  = [1]
			d_par_min = [2]
			k_per_min = [3]
			S0_min    = [4]

			ang1_max  = [0]
			ang2_max  = [1]
			d_par_max = [2]
			k_per_max = [3]
			S0_max    = [4]


			self.param_min = np.array([ang1_min, ang2_min, d_par_min, k_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_par_max, k_per_max, S0_max])


	def forward(self, x):
		''' Full forward pass of a qMRI-Net, estimating MRI signals given input MRI measurements

		    y = mynet.forward(x) 
		    y = mynet(x) 

		    * mynet:  initialised qMRI-Net

		    * x:      pytorch Tensor storing a set of MRI measurements from one voxel or from a mini-batch
			      (for a mini-batch of multiple voxels, x has size voxels x measurements)

		    * y:      pytorch Tensor storing a prediction of x (it has same size as x)
		'''	

		encoded2 = self.getparams(x)          # From MRI measurements --> to tissue parameters
		decoded = self.getsignals(encoded2)   # From tissue parameters --> to predicted MRI signals 

		return decoded

    




