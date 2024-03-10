
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch import exp as texp 
from torch import reshape as tshape
from torch import matmul as tmat
'''
This network is used in the syntheiss of signals from some input parameters.
Input for definition:
	- model (for now only Zeppelin is implemented)
	- bvals
	- bvecs
	
It consists of two def:
	- synthesize_signals: inut parameters. output synthesited signals
	- changelim: allows change limits of parameters
'''

class network_synsignals(nn.Module): 

	# Constructor
	def __init__(self, mrimodel, bvals, g):

		super(network_synsignals, self).__init__()
	
		self.mrimodel = mrimodel                    # MRI signal model. In this case Zeppelin
		self.bvals = np.array(bvals)                # MRI protocol. bvals
		self.bvecs = g                                  # gradient directions
		
		
		if self.mrimodel=='Zeppelin':   # Parameter order:  ang1, ang2, AD, RD, S0

			S0_min    = 0.0
			d_par_min = 0.0
			k_per_min = 0.0
			ang1_min  = 0.0
			ang2_min  = 0.0 


			S0_max    = 5
			d_par_max = 3.2
			k_per_max = 1
			ang1_max  = np.pi
			ang2_max  = 2*np.pi


			self.param_min = np.array([ang1_min, ang2_min, d_par_min, k_per_min, S0_min]) 
			self.param_max = np.array([ang1_max, ang2_max, d_par_max, k_per_max, S0_max])
			self.param_name =  ['ang1', 'ang2', 'd_par', 'k_per', 'S0']


	def synthesize_signals(self, x):
		
		if self.mrimodel =='Zeppelin':  # Parameter order:  ang1, ang2, AD, RD, S0
	        
			b_delta = 1
			
			# Repeat bvalues Nmeas times
			bval = Tensor(np.tile(self.bvals, (x.shape[0], 1)))                  # Shape: (N, 108)
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
	
	
	def changelim(self, pmin, pmax):
		'''Change limits of tissue parameters
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
	
			param_min = np.array([ang1_min, ang2_min, d_z_par_min, d_z_per_min, S0_min]) 
			param_max = np.array([ang1_max, ang2_max, d_z_par_max, d_z_per_max, S0_max])
			
			
			
			
			