
### Load libraries
import argparse, os, sys 
import pickle as pk
import numpy as np
from torch import Tensor
from pathlib import Path as pt
sys.path.insert(0, os.path.dirname(pt(__file__).absolute()) )
import network_synsignals
import time

'''
	Output saved:
		
		- file *_sigtrain.bin will store synthetic training MRI signals as a numpy matrix (rows: voxels; columns: measurements); Noised. 
		- file *_sigval.bin will store synthetic validation MRI signals as a numpy matrix (rows: voxels; columns: measurements); Noised.
		- file *_sigtest.bin will store synthetic test MRI signals as a numpy matrix (rows: voxels; columns: measurements); Noised.
		
		- file *_GT_sigtrain.bin will store GT synthetic MRI signals as a numpy matrix (rows: voxels; columns: measurements) 
		- file *_GT_sigval.bin will store GT synthetic MRI signals as a numpy matrix (rows: voxels; columns: measurements)
		- file *_GT_sigtest.bin will store GT synthetic test MRI signals as a numpy matrix (rows: voxels; columns: measurements)
		
		- file *_GT_paramtrain.bin will store GT tissue parameters corresponding to training MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); 
		- file *_GT_paramval.bin will store GT tissue parameters corresponding to validation MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); 
		- file *_GT_paramtest.bin will store GT tissue parameters corresponding to test MRI signals as a numpy matrix (rows: voxels; columns: tissue parameters); 
		
'''


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='This program synthesise MRI signals. Modified version: Paula Castro. Based on: Author: Francesco Grussu, University College London ')
	parser.add_argument('mri_model', 
					 help='string indicating the MRI model to fit (choose among: "Zeppelin", for dMRI data without TE and T2 variations. Parameters: [ang(1), ang(2), d_z_par, d_z_per, S0]' )
	parser.add_argument('bvals', 
					 help='path to text file storing the desired b values. b-values in s/mm^2')
	parser.add_argument('bvecs', 
					 help='path to text file storing the desired gradient directions')
	parser.add_argument('out_str', 
					 help='base name of output files, to which various strings will be appended.')
	parser.add_argument('--ntrain', metavar='<value>', default='8000', 
					 help='number of synthetic voxels for training (default: 8000)')
	parser.add_argument('--nval', metavar='<value>', default='2000', 
					 help='number of synthetic voxels for validation (default: 2000)')	
	parser.add_argument('--ntest', metavar='<value>', default='1000', 
					 help='number of synthetic voxels for testing (default: 1000)')
	parser.add_argument('--snr', metavar='<value>', default='70', 
					 help='value for desired snr to be used to corrupt the data with synthetic noise; default 70)')	
	parser.add_argument('--noise', metavar='<value>', default='gauss', 
					 help='synthetic noise type (choose among "gauss", "rician" and "noncentral_chisquare" ; (default: gauss)')
	parser.add_argument('--seed', metavar='<value>', default='20180721', 
					 help='seed for random number generation (default: 20180721)')	
	parser.add_argument('--bias', metavar='<value>', default='100', 
					 help='multiplicative constant used to scale all synthetic signals, representing the offset proton density signal level with no weigthings (default: 100.0)')
	parser.add_argument('--parmin', metavar='<value>', 
					 help='list of lower bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 0.0,,0.0,0.0,0.0,0.0 for model Zeppelin)')
	parser.add_argument('--parmax', metavar='<value>', 
					 help='list of upper bounds of tissue parameters. Entries corresponding to different parameters should be separated by a comma (for example: 3.14,,6.28,3.2,1,5 for model Zeppelin).')	
	args = parser.parse_args()

	### Get input parameters
	outstr = args.out_str                          # extract output string
	mrimodel = args.mri_model                      # extract mri model selected
	bvals = args.bvals                             # extract bvalues path
	bvecs = args.bvecs                             # extract gradient directions path

	ntrain = int(args.ntrain)                      # number training samples
	nval = int(args.nval)                          # number val samples
	ntest = int(args.ntest)                        # number test
	
	snr = int(args.snr)                            # snr value

	noisetype = args.noise                         # extract noise type
	myseed = int(args.seed)                        # extract seed  
	bsig = float(args.bias)                        # extract bias


	### Get optional modified bounds for tissue parameters
	if (args.parmin is not None) or (args.parmax is not None):
		if (args.parmin is not None) and (args.parmax is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
		if (args.parmax is not None) and (args.parmin is None):
			raise RuntimeError('you need to set both parmin and parmax options simultaneously')
					
		# Lower bound
		pminbound = (args.parmin).split(',')                     # extract the lower bounds if specified
		pminbound = np.array( list(map( float, pminbound )) )
		
		# Upper bound
		pmaxbound = (args.parmax).split(',')                    #  extract the upper bounds if specified
		pmaxbound = np.array( list(map( float, pmaxbound )) )
			

	### Print information
	print('\n\nSYNTHESISE DATA NUMERICALLY                 ')
	print(f'\n** Output files : {args.out_str}')
	print(f'** MRI model: {args.mri_model}')
	print(f'** MRI protocol file: {args.bvals}')
	print(f'                      {args.bvecs}')

	print(f'** Noise type: {args.noise}')
	print(f'** SNR levels: {snr}')
	print(f'** Seed for random number generators: {myseed}')
	print(f'** Unweighted signal bias: {bsig}')

	start_time = time.time()
	
	### Load MRI protocol: bvals and bvecs
	
	bvals = np.loadtxt(args.bvals)
	bvals = np.array(bvals)/1000.0   # to convert to correct units!!
	g = np.loadtxt(args.bvecs)
	
	# For Zeppein simulations we dont need that many measuremnts. 108 are enough
	
	bvals = bvals[:108]
	g = g[:108,:]
	
	Nmeas = bvals.shape
	
	print('\n** Nmeas per signal: ', Nmeas)
	
	### Check that MRI model exists. For now we only have the Zeppelin implemented
	if ( (mrimodel!='Zeppelin')):
		raise RuntimeError('The chosen MRI model is not implemented. Choose among: Zeppelin')

	### Check that the noise type exists
	if ( (noisetype!='rician') and (noisetype!='gauss') and (noisetype!='noncentral_chisquare')):
		raise RuntimeError('The chosen noise model is not implemented. Choose among: gauss, rician or noncentral_chisquare ')
    
	### MODEL
	if mrimodel=='Zeppelin':
		totpar = 5                 # ang1, ang2, AD, RD, S0
		s0idx= 4                   # S0 is in position 4
		
		network = network_synsignals.network_synsignals(mrimodel,bvals, g)

	if (args.parmin is not None) or (args.parmax is not None):
		network.changelim(pminbound,pmaxbound)
    
	print(f'\n** Tissue parameter lower bounds: {network.param_min}')	
	print(f'** Tissue parameter upper bounds: {network.param_max}')	
	
	### Set random seeds
	np.random.seed(myseed)       # Random seed for reproducibility: NumPy

	print('\n                 ... synthesising uniformly-distributed tissue parameters')
	
	npars = network.param_min.size    # number of parameters to synthesise from the model selected 
	
	ptrain = np.zeros((ntrain,npars))   # matrix with ntrain number of rows and npars number of cols
	pval = np.zeros((nval,npars))
	ptest = np.zeros((ntest,npars))
    
	### Create output folder
	out_base_dir = '{}_MRImodel{}_ntrain{}_nval{}_ntest{}_SNR{}'.format(outstr, mrimodel, ntrain, nval, ntest,snr)
	if(os.path.isdir(out_base_dir)==False):	
		os.mkdir(out_base_dir) 

	outstr = os.path.join(out_base_dir , 'syn')
    
	print('\n** Folder created at: ', outstr)
    
	for pp in range(0, npars,1):		
		ptrain[:,pp] = network.param_min[pp] + (network.param_max[pp] - network.param_min[pp])*np.random.rand(ntrain) 
		pval[:,pp] = network.param_min[pp] + (network.param_max[pp] - network.param_min[pp])*np.random.rand(nval) 
		ptest[:,pp] = network.param_min[pp] + (network.param_max[pp] - network.param_min[pp])*np.random.rand(ntest) 
        
	print('\nshape ptest: ' , ptest.shape)
	print('shape ptrain: ', ptrain.shape)
	print('shape pval: '  , pval.shape)


	print(f'\n                 ... generating noise-free signals with {Nmeas} measurements')
    
	# Call the synthesize_signals() method of NETWORK, passing the parameters as tensors 
	strain = network.synthesize_signals(Tensor(ptrain))
	strain = strain.detach().numpy()     
	print('\nstrain: ', strain.shape)
	strain = int(bsig) * strain          # multiply by bsig (the bias. scaling factor)

	sval = network.synthesize_signals(Tensor(pval))
	sval = sval.detach().numpy()
	print('sval: ', sval.shape)
	sval = int(bsig)*sval

	stest = network.synthesize_signals(Tensor(ptest))
	stest = stest.detach().numpy()
	print('stest: ', stest.shape)
	stest = int(bsig)*stest
	    

	print(f'\n                 ... generating noised signals with {Nmeas} measurements')
    
	print('\nValue of snr: ', snr)
	print('Noise type: ', noisetype)
	
	
	# three types of noise are implemented: gauss, rician and non central chi squared

	def gaussian_noise(signal, snr):
	    s0 = np.mean(signal) # mean signal intensity of the image
	    sigma_noise = s0 / snr

	    gaussian_noise = np.random.normal(scale=sigma_noise, size=signal.shape) # Generate Gaussian noise
	    noisy_image = signal + gaussian_noise
		
	    return noisy_image


	def rician_noise(signal, snr):
	    s0 = np.mean(signal**2) # mean signal intensity of the image
	    sigma_noise = s0 / snr

	    gaussian_noise = np.random.normal(scale=np.sqrt(sigma_noise / 2), size=signal.shape) # Gaussian noise
	    rayleigh_noise = np.random.rayleigh(scale=np.sqrt(sigma_noise / (4 - np.pi)), size=signal.shape) # Rayleigh-distributed noise
	    rician_noise = np.sqrt(gaussian_noise**2 + rayleigh_noise**2) # combine them
	    noisy_image = signal + rician_noise
		
	    return noisy_image


	def noncentral_chisquare_noise(signal, snr):
	    s0 = np.mean(signal**2) # mean signal intensity of the image
	    sigma_noise = s0 / snr

	    gaussian_noise = np.random.normal(scale=np.sqrt(sigma_noise), size=signal.shape) #  Gaussian noise
	    noncentral_chisquare_noise = np.random.noncentral_chisquare(df=2, nonc=10, size=signal.shape) # non-central chi-square noise
	    noncentral_chisquare_noise = np.sqrt(gaussian_noise**2 + noncentral_chisquare_noise) # combine
	    noisy_image = signal + noncentral_chisquare_noise
		
	    return noisy_image
	

	if noisetype == 'noncentral_chisquare':
	    strain_noised = noncentral_chisquare_noise(strain, snr)
	    sval_noised   = noncentral_chisquare_noise(sval, snr)
	    stest_noised  = noncentral_chisquare_noise(stest, snr)

	if noisetype == 'gauss':
	    strain_noised = gaussian_noise(strain, snr)
	    sval_noised   = gaussian_noise(sval, snr)
	    stest_noised  = gaussian_noise(stest, snr)
		
	if noisetype == 'rician':
	    strain_noised = rician_noise(strain, snr)
	    sval_noised   = rician_noise(sval, snr)
	    stest_noised  = rician_noise(stest, snr)
		
	print('\nstrain noised: ', strain_noised.shape)
	print('sval noised: '    , sval_noised.shape)
	print('stest noised: '   , stest_noised.shape)

                
	### Save output files
	print('\n                 ... saving output files')
        
	try:
        
        # save noisy sigals
		my_file = open('{}_sigtrain.bin'.format(outstr),'wb')
		pk.dump(strain_noised,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigval.bin'.format(outstr),'wb')
		pk.dump(sval_noised,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_sigtest.bin'.format(outstr),'wb')
		pk.dump(stest_noised,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

        
        # save GT sigals
		my_file = open('{}_GT_sigtrain.bin'.format(outstr),'wb')
		pk.dump(strain,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_GT_sigval.bin'.format(outstr),'wb')
		pk.dump(sval,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_GT_sigtest.bin'.format(outstr),'wb')
		pk.dump(stest,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()


        # save GT parameters
		my_file = open('{}_GT_paramtrain.bin'.format(outstr),'wb')
		pk.dump(ptrain,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_GT_paramval.bin'.format(outstr),'wb')
		pk.dump(pval,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

		my_file = open('{}_GT_paramtest.bin'.format(outstr),'wb')
		pk.dump(ptest,my_file,pk.HIGHEST_PROTOCOL)
		my_file.close()

	except: 		
		raise RuntimeError('The output folder may not exist or you may lack permission to write there!')

	### Done
	end_time = time.time()

	print('\nExecution time: ', end_time - start_time, ' seconds')
    
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
				file.write('  ** g: ' + str(g.shape) + '\n'+ '\n')

				file.write('Execution time: ' + str(end_time - start_time) + ' seconds'+ '\n')


		print('')
		print("INFO has been saved to the file.")
        
	except: 		
		raise RuntimeError('the output folder may not exist or you may lack permission to write there!')
        

	print('\n                 ... done!')
	print('')
	sys.exit(0)


	
