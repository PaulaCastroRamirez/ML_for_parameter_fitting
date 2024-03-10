
### Load libraries
import argparse, os, sys 
import pickle as pk
import numpy as np
import torch
from torch import Tensor
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import network
import time


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program synthesise MRI signals that can be used to train a qMRI-net for quantitative MRI parameter estimation.  Author: Francesco Grussu, University College London (<f.grussu@ucl.ac.uk><francegrussu@gmail.com>). Code released under BSD Two-Clause license. Copyright (c) 2020 University College London. All rights reserved.')
	parser.add_argument('mri_model', help='string indicating the MRI model to fit (choose among: "Zeppelin", for dMRI data without TE and T2 variations. Parameters: [ang(1), ang(2), d_z_par, d_z_per, S0]' )
	parser.add_argument('mri_prot', help='path to text file storing the MRI protocol. For model "Zeppelin" it must contain a matrix where the 1st row stores b-values in s/mm^2, NO SECOND ROW NOW (while 2nd row echo times in ms)')
	parser.add_argument('out_str', help='base name of output files, to which various strings will be appended: file *_sigtrain.bin will store synthetic training MRI signals as a numpy matrix (rows: voxels; columns: measurements); file *_sigval.bin will store synthetic validation MRI signals as a numpy matrix (rows: voxels; columns: measurements); file *_sigtest.bin will store synthetic test MRI signals as a numpy matrix (rows: voxels; columns: measurements); file *_paramtrain.bin will store tissue parameters corresponding to training MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); file *_paramval.bin will store tissue parameters corresponding to validation MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); file *_paramtest.bin will store tissue parameters corresponding to test MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); file *_sigmatrain.bin will store voxel-wise noise levels for validation voxels as a numpy matrix (rows: voxels; columns: 1); file *_sigmaval.bin will store voxel-wise noise levels for validation voxels as a numpy matrix (rows: voxels; columns: 1); file *_sigmatest.bin will store voxel-wise noise levels for test voxels as a numpy matrix (rows: voxels; columns: 1)')
	parser.add_argument('g', help='path to text file storing the gradient directions')
	parser.add_argument('--ntrain', metavar='<value>', default='100', help='number of synthetic voxels for training (default: 8000; note that if multiple SNR levels are requested, the final number of training voxels may change slightly from what has been requested)')
	parser.add_argument('--nval', metavar='<value>', default='50', help='number of synthetic voxels for validation (default: 2000; note that if multiple SNR levels are requested, the final number of training voxels may change slightly from what has been requested)')	
	parser.add_argument('--ntest', metavar='<value>', default='20', help='number of synthetic voxels for testing (default: 1000; note that if multiple SNR levels are requested, the final number of training voxels may change slightly from what has been requested)')
	parser.add_argument('--snr', metavar='<value>', default='100', help='value or values (separated by hyphens) of signal-to-noise ratio to be used to corrupt the data with synthetic noise (example: --snr 20 or --snr 20-15-10; default 30; values higher than 1e6 will be mapped to infinity, i.e. no noise added). SNR is evaluated with respect to non-weighted signals (e.g. signal with no inversion pulses, no diffusion-weighting, minimum TE etc)')	
	parser.add_argument('--noise', metavar='<value>', default='gauss', help='synthetic noise type (choose among "gauss", "rician" and "noncentral_chisquare" ; default "gauss")')
	parser.add_argument('--seed', metavar='<value>', default='20180721', help='seed for random number generation (default: 20180721)')	
	parser.add_argument('--bias', metavar='<value>', default='100', help='multiplicative constant used to scale all synthetic signals, representing the offset proton density signal level with no weigthings (default: 100.0)')
	parser.add_argument('--parmin', metavar='<value>', help='list of lower bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 0.5,0.2,250,0.5 for model br_sirsmdt). Tissue parameters are: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0; model "twocompdwite", parameters v, Da, t2a, Db, Kb, t2b, s0, where a and b indicate compartments a and b. If not specified, default tissue parameter ranges are used.')
	parser.add_argument('--parmax', metavar='<value>', help='list of upper bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 2.4,0.9,3000,5.0 for model br_sirsmdt). Tissue parameters are: model "pr_hybriddwi", parameters vl, v s.t. ve=(1-vl)*v, Dl, De, Ds, t2l, t2e, t2s, s0, where l/e/stroma stands for lumen/epithelium/stroma; model "br_sirsmdt", parameters dpar, kperp s.t. dperp=kperp*dpar, t1, s0; model "twocompdwite", parameters v, Da, t2a, Db, Kb, t2b, s0, where a and b indicate compartments a and b. If not specified, default tissue parameter ranges are used.')	
	args = parser.parse_args()

	### Get input parameters
	outstr = args.out_str                          # extract output string
	mrimodel = args.mri_model                      # extract mri model selected
	g = args.g                                     # extract mri model selected

	ntrain = int(args.ntrain)                      # optional argument
	nval = int(args.nval)                          # optional argument
	ntest = int(args.ntest)                        # optional argument
	snr = (args.snr).split('-')                    # optional argument
	snr = np.array( list(map( float,snr )) )       
	snr[snr>1e6] = np.inf     
	nsnr = snr.size    
	noisetype = args.noise                         # optional argument
	myseed = int(args.seed)                        # optional argument   
	bsig = float(args.bias)                        # optional argument 

	### Make sure the number of training/test/validation voxels can be divided by the number of SNR levels
	ntrain = int(np.round(float(ntrain)/float(nsnr))*nsnr)
	nval = int(np.round(float(nval)/float(nsnr))*nsnr)
	ntest = int(np.round(float(ntest)/float(nsnr))*nsnr)
	
	### Get optional user-defined bounds for tissue parameters
	if (args.parmin is not None) or (args.parmax is not None):
		
		if (args.parmin is not None) and (args.parmax is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
		
		if (args.parmax is not None) and (args.parmin is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
					
		# Lower bound
		pminbound = (args.parmin).split(',')                     # extract the bounds if specified
		pminbound = np.array( list(map( float, pminbound )) )
		
		# Upper bound
		pmaxbound = (args.parmax).split(',')
		pmaxbound = np.array( list(map( float, pmaxbound )) )
			

	### Print some information
	print('')
	print('')
	print('********************************************************************')
	print('                 SYNTHESISE DATA TO TRAIN THE NETWORK                ')
	print('********************************************************************')
	print('')
	print('** Output files : {}_*'.format(args.out_str))
	print('** MRI model: {}'.format(args.mri_model))
	print('** MRI protocol file: {}'.format(args.mri_prot))
	print('** Noise type: {}'.format(args.noise))
	print('** SNR levels: {}'.format(snr))
	print('** Seed for random number generators: {}'.format(myseed))
	print('** Unweighted signal bias: {}'.format(bsig))

	start_time = time.time()
	### Load MRI protocol
	try:
        
		bvals = []
		TEs = None

        # Open the file for reading
		with open(args.mri_prot, 'r') as file:
            # Read the contents of the file
				file_contents = file.readlines()
				bvals = [float(num) for num in file_contents[0].split()]

            # Check if there are at least two rows in the file
# =============================================================================
# 		if len(file_contents) >= 2:
#                 # Process the first row (numbers separated by spaces)
# 				bvals = [float(num) for num in file_contents[0].split()]
# 
# 				TEs = [float(num) for num in file_contents[1].split()]
# 				TEs = np.array(TEs)
# =============================================================================

        
# =============================================================================
# 		print("First Row Numbers:", bvals)
# 		print("Second Row Number:", TEs)
# =============================================================================
                                
	except: 		
		raise RuntimeError('the MRI protocol is not in a suitable text file format. Sorry!')

	bvals = np.array(bvals)/1000.0
	g = np.loadtxt(args.g)

	Nmeas = bvals.size
	print('Nmeas: ', Nmeas)

	mriprot = [bvals,TEs]
	mriprot = np.array(mriprot, dtype=object)
    
	#print('mriprotocol: {}'.format(mriprot))	
	print('shape mri protocol: ', mriprot.shape)
	#print('mri protocol: ', mriprot)


	### Check that MRI model exists
	if ( (mrimodel!='Zeppelin')):
		raise RuntimeError('the chosen MRI model is not implemented. Sorry!')

	### Check that the noise model exists
	if ( (noisetype!='rician') and (noisetype!='gauss') and (noisetype!='noncentral_chisquare')):
		raise RuntimeError('the chosen noise model is not implemented. Sorry!')
    
	### Get a qMRI-net
	if mrimodel=='Zeppelin':
		totpar = 5
		s0idx= 4 # because of the order of our paramters, S0 is in position 0
		qnet = network.mrisig([Nmeas,totpar],0.0,'Zeppelin',mriprot, g)

	if (args.parmin is not None) or (args.parmax is not None):
		qnet.changelim(pminbound,pmaxbound)
    
	print('** Tissue parameter lower bounds: {}'.format(qnet.param_min))	
	print('** Tissue parameter upper bounds: {}'.format(qnet.param_max))	

	### Set random seeds
	np.random.seed(myseed)       # Random seed for reproducibility: NumPy
	torch.manual_seed(myseed)    # Random seed for reproducibility: PyTorch

	### Synthesise uniformly-distributed tissue parameters
	print('')
	print('                 ... synthesising tissue parameters')
	npars = qnet.param_min.size    # number of parameters to synthesise from the model selected 
	print('value npars: ', npars)

	ptrain = np.zeros((ntrain,npars))   # matrix with ntrain number of rows and npars number of cols
	pval = np.zeros((nval,npars))
	ptest = np.zeros((ntest,npars))

    # These three lines populate the ptrain, pval, and ptest arrays with randomly generated 
    # values within a specified range for each tissue parameter 'pp'
    # The random values are generated using np.random.rand() function, which returns an array of random 
    # numbers between 0 and 1. The generated random values are scaled by the range between qnet.param_min[pp]
    # and qnet.param_max[pp] and then added to qnet.param_min[pp] to ensure the generated values fall 
    # within the desired range
    
	### Create output base name
	out_base_dir = '{}_MRImodel{}_ntrain{}_nval{}_ntest{}_SNR{}'.format(outstr, mrimodel,ntrain,nval,ntest,snr)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir) # create the folder if it does not exist

	outstr = os.path.join(out_base_dir , 'HCP')
    
	print('** Folder created: ', outstr)
    
	for pp in range(0, npars,1):		
		ptrain[:,pp] = qnet.param_min[pp] + (qnet.param_max[pp] - qnet.param_min[pp])*np.random.rand(ntrain) 
		pval[:,pp] = qnet.param_min[pp] + (qnet.param_max[pp] - qnet.param_min[pp])*np.random.rand(nval) 
		ptest[:,pp] = qnet.param_min[pp] + (qnet.param_max[pp] - qnet.param_min[pp])*np.random.rand(ntest) 
        
	print('shape ptest: ' , ptest.shape)
	print('shape ptrain: ', ptrain.shape)
	print('shape pval: '  , pval.shape)


	### Predict noise-free signals
	print('')
	print('                 ... generating noise-free signals ({} measurements)'.format(Nmeas))
    
	# invoke the getsignals() method of NETWORK object, passing Tensor(ptrain) as an argument. 
	# ptrain is a matrix or array containing training signal samples. The getsignals() method 
	# returns the predicted signals corresponding to the input parameters, and the result is 
    # assigned to the variable strain.
        
	#[strain, g_train, bvals_train] = qnet.getsignals(Tensor(ptrain))
	strain = qnet.getsignals(Tensor(ptrain))

	strain = strain.detach().numpy()     # detach the strain tensor from the computational graph and converts it to a NumPy array.
        
	print('strain: ', strain.shape)
	strain = int(bsig) * strain       # multiply by bsig (the bias. scaling factor)

	#[sval, g_val, bvals_val] = qnet.getsignals(Tensor(pval))
	sval = qnet.getsignals(Tensor(pval))

	print('sval: ', sval.shape)

	sval = sval.detach().numpy()
	sval = int(bsig)*sval

	#[stest, g_test, bvals_test] = qnet.getsignals(Tensor(ptest))
	stest = qnet.getsignals(Tensor(ptest))

	print('stest: ', stest.shape)

	stest = stest.detach().numpy()
	stest = int(bsig)*stest
    
	print('strain', strain.shape)

	### Generate arrays of sigma of noise levels
	print('') 
	print('                 ... Adding noise')
    
	print('value of snr: ', snr)
	print('value of nsnr: ', nsnr)

# =============================================================================
#     # save the protocol
# 
# 	my_file = open('{}_bvals_train.bin'.format(outstr),'wb')
# 	pk.dump(bvals_train,my_file,pk.HIGHEST_PROTOCOL)
# 	my_file.close()
#     
# 	my_file = open('{}_g_train.bin'.format(outstr),'wb')
# 	pk.dump(g_train,my_file,pk.HIGHEST_PROTOCOL)
# 	my_file.close()
#         
# 	my_file = open('{}_TEs_train.bin'.format(outstr),'wb')
# 	pk.dump(TEs_train,my_file,pk.HIGHEST_PROTOCOL)
# 	my_file.close()
# =============================================================================
    
    
    # initialize empty arrays with zero rows and one column. They will be used to store the signal-to-noise 
    # ratio (SNR) levels for the training, validation, and test data
	snr_train = np.ones((0,1))
	snr_val = np.ones((0,1))
	snr_test = np.ones((0,1))
    
	for ss in range(0,nsnr):
		snr_train = np.concatenate(  ( snr_train , snr[ss]  *np.ones((int(ntrain/nsnr),1)) ) , axis = 0 ) 
		snr_val = np.concatenate  (  ( snr_val ,   snr[ss]  *np.ones((int(nval/nsnr),1)) )   , axis = 0 ) 
		snr_test = np.concatenate (  ( snr_test ,  snr[ss]  *np.ones((int(ntest/nsnr),1)) )  , axis = 0 ) 
        
	#print('value of snr_train, snr_val and snr_test: ', snr_train, snr_val, snr_test)

	# shuffle the values in the snr_train, snr_val, and snr_test arrays randomly. 
	# This is done using the np.random.shuffle() function to ensure the SNR levels are assigned randomly 
    # to the training, validation, and test samples.
	np.random.shuffle(snr_train)
	np.random.shuffle(snr_val)
	np.random.shuffle(snr_test)

	# compute the noise standard deviation. Divide the scaled signal by the SNR to get sigma value
	sgm_train = (bsig*(ptrain[:,s0idx:s0idx+1])) / snr_train

	sgm_val = (bsig*pval[:,s0idx:s0idx+1]) / snr_val
	sgm_test = (bsig*ptest[:,s0idx:s0idx+1]) / snr_test
    
	#print('ptrain[:,s0idx:s0idx+1] shape: ', ptrain[:,s0idx:s0idx+1])
	#print('sgm_test: ', sgm_test)
	#print('stest: ', stest[:,s0idx:s0idx+1])


	print('sgm_train shape: ', sgm_train.shape)
	print('sgm_val shape: ', sgm_val.shape)
	print('sgm_test shape: ', sgm_test.shape)
	print('') 

	### Add noise
    
	# initialize with zeros. The shape of each array matches the shape of the corresponding original signal array.
	# these arrays will contain the noisy versions of the signals.
	strain_noisy = np.zeros(strain.shape)
	sval_noisy = np.zeros(sval.shape)
	stest_noisy = np.zeros(stest.shape)
    
	print('strain_noisy.shape: ', strain_noisy.shape)
	print('sval_noisy.shape: ', sval_noisy.shape)
	print('stest_noisy.shape: ', stest_noisy.shape)

    
# =============================================================================
# 	def get_dimensions(lst):
# 		dimensions = []
# 		while isinstance(lst, list):
# 			dimensions.append(len(lst))
# 			lst = lst[0] if len(lst) > 0 else None
# 		return dimensions
#     
# 	dimensions_strain = get_dimensions(strain)
# 	print('strain dimensions: ', dimensions_strain)
# 	print('value ntrain: ', ntrain)
# 	print('') 
# =============================================================================


	for tt in range(0, np.array(ntrain), 1):
        
		if (noisetype== 'noncentral_chisquare'):
            
			noise = np.random.noncentral_chisquare(10 * np.random.rand(strain.shape[1],), 10 * np.random.rand(strain.shape[1],))
			#print('noise / np.std(noise)', noise / np.std(noise))

			noise = sgm_train[tt] * (noise / np.std(noise))           # scale the standard deviation
			#print('noise', noise)

			strain_noisy[tt,:] = strain[tt] + noise

		if(noisetype=='gauss'):
            
			noise = sgm_train[tt]*np.random.randn(strain.shape[1],) 
			noise = sgm_train[tt] * (noise / np.std(noise)) 
			#print('noise', noise)

			strain_noisy[tt,:] = strain[tt,:] + noise 
    
		if(noisetype=='rician'):
            
			noise1 = sgm_train[tt]*np.random.randn(strain.shape[1],)
			noise2 = ( sgm_train[tt]*np.random.randn(strain.shape[1],) )**2 
			noise1 = sgm_train[tt] * (noise1 / np.std(noise1)) 
			noise2 = sgm_train[tt] * (noise2 / np.std(noise2)) 

			strain_noisy[tt,:] = np.sqrt((strain[tt,:] + noise1)**2 + noise2)
            
        
	for vv in range(0, np.array(nval), 1):
        
		if (noisetype== 'noncentral_chisquare'):
            
			noise = np.random.noncentral_chisquare(10 * np.random.rand(sval.shape[1],), 10 * np.random.rand(sval.shape[1],))
			noise = sgm_val[vv] * (noise / np.std(noise))           # scale the standard deviation
			sval_noisy[vv,:] = sval[vv] + noise

		if(noisetype=='gauss'):
            
			noise = sgm_val[vv]*np.random.randn(sval.shape[1],) 
			noise = sgm_val[vv] * (noise / np.std(noise)) 
			sval_noisy[vv,:] = sval[vv,:] + noise 
    
		if(noisetype=='rician'):
            
			noise1 = sgm_val[vv] * np.random.randn(sval.shape[1],)
			noise2 = (sgm_val[vv]* np.random.randn(sval.shape[1],) )**2 
			noise1 = sgm_val[vv] * (noise1 / np.std(noise1)) 
			noise2 = sgm_val[vv] * (noise2 / np.std(noise2)) 

			sval_noisy[vv,:] = np.sqrt((sval[vv,:] + noise1)**2 + noise2)

	for qq in range(0, np.array(ntest), 1):
        
		if (noisetype== 'noncentral_chisquare'):
            
			noise = np.random.noncentral_chisquare(10 * np.random.rand(stest.shape[1],), 10 * np.random.rand(stest.shape[1],))
			noise = sgm_test[qq] * (noise / np.std(noise))           # scale the standard deviation
			stest_noisy[qq,:] = stest[qq] + noise
			#print('noise', (noise / np.std(noise)))

		if(noisetype=='gauss'):
            
			noise = sgm_test[qq]*np.random.randn(stest.shape[1],) 
			noise = sgm_test[qq] * (noise / np.std(noise)) 
			stest_noisy[qq,:] = stest[qq,:] + noise 
    
		if(noisetype=='rician'):
            
			noise1 = sgm_test[qq] * np.random.randn(stest.shape[1],)
			noise2 = (sgm_test[qq]* np.random.randn(sval.shape[1],) )**2 
			noise1 = sgm_test[qq] * (noise1 / np.std(noise1)) 
			noise2 = sgm_test[qq] * (noise2 / np.std(noise2)) 

			stest_noisy[qq,:] = np.sqrt((stest[qq,:] + noise1)**2 + noise2)    			
                

	### Save output files
	print('')
	print('                 ... saving output files')
	#print('strain_noisy size: ', strain_noisy)
	print('sgm_train size: ', sgm_train.shape)
	print('Nmeas value (it is an int) ', Nmeas)
        
	print(np.sum(sval_noisy))

	try:
        
        # save noisy sigals
		my_file = open('{}_sigtrain.bin'.format(outstr),'wb')
		pk.dump(strain_noisy,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigval.bin'.format(outstr),'wb')
		pk.dump(sval_noisy,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigtest.bin'.format(outstr),'wb')
		pk.dump(stest_noisy,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

        
        # save no noisy sigals
		my_file = open('{}_sigtrain_NOnoise.bin'.format(outstr),'wb')
		pk.dump(strain,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigval_NOnoise.bin'.format(outstr),'wb')
		pk.dump(sval,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigtest_NOnoise.bin'.format(outstr),'wb')
		pk.dump(stest,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()



        # save synthesited parameters

		my_file = open('{}_paramtrain.bin'.format(outstr),'wb')
		pk.dump(ptrain,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_paramval.bin'.format(outstr),'wb')
		pk.dump(pval,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_paramtest.bin'.format(outstr),'wb')
		pk.dump(ptest,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()


        # save SNR values of the signals
		my_file = open('{}_sigmatrain.bin'.format(outstr),'wb')
		pk.dump(sgm_train,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigmaval.bin'.format(outstr),'wb')
		pk.dump(sgm_val,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigmatest.bin'.format(outstr),'wb')
		pk.dump(sgm_test,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()


	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')

	### Done
	end_time = time.time()
	print('') 

	print('Execution time: ', end_time - start_time, ' seconds')
    
	try:
        # File path
		file_path = outstr + '_INFO.txt'

		with open(file_path, 'w') as file:
				file.write('strain shape: ' + str(strain.shape) + '\n')
				file.write('sval shape: ' + str(sval.shape) + '\n')
				file.write('stest shape: ' + str(stest.shape) + '\n' + '\n')

				file.write('SNR: ' + str(snr) + '\n')
				file.write('noise type: ' + str(noisetype)+ '\n')
				file.write('Bias: ' + str(bsig) + '\n' + '\n')
				file.write('MRI model: ' + str(mrimodel) + '\n')
				file.write('MRI protocol: ' + '\n')
				file.write('  ** bvals: ' + str(bvals.shape) + '\n')
				file.write('  ** TEs: ' + str(TEs) + '\n')

				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')
        

	print('')
	print('                 ... done!')
	print('')
	sys.exit(0)


	
