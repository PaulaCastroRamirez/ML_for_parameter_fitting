import numpy as np
import math
from numpy import matlib

def vec(x, mask=None):
    """
    Converts a 4D xyzn matrix with n the number of measurements per voxel
    to a 2D nm matrix with m the number of voxels with mask == True.

    Args:
    x (numpy.ndarray): Input 4D matrix.
    mask (numpy.ndarray, optional): Mask for selecting specific voxels.

    Returns:
    y (numpy.ndarray): Output 2D matrix.
    mask (numpy.ndarray): Updated mask if not provided.
    """
    if mask is None:
        mask = ~np.isnan(x[:, :, :, 0])

    num_measurements = x.shape[3]
    selected_voxels = mask.sum()
    
    y = np.zeros((num_measurements, selected_voxels), dtype=x.dtype)

    for k in range(num_measurements):
        Dummy = x[:, :, :, k]
        y[k, :] = Dummy[mask]

    return y, mask


def unvec(x, mask):
    """
    Converts a 2D nm matrix with n the number of measurements per voxel
    and m the number of voxels with mask == True, to a 4D xyzn matrix.

    Args:
    x (numpy.ndarray): 2D matrix with shape (m, n).
    mask (numpy.ndarray): 3D boolean mask with shape (x_dim, y_dim, z_dim).

    Returns:
    numpy.ndarray: 4D matrix with shape (x_dim, y_dim, z_dim, n).
    """

    # Get the dimensions of the mask
    dims = mask.shape

    # Create an empty 4D array filled with NaN values
    y = np.full((dims[0], dims[1], dims[2], x.shape[0]), np.nan, dtype=x.dtype)

    for k in range(x.shape[0]):
        # Create a Dummy array with the same dimensions as the mask
        dummy = np.full(dims, np.nan, dtype=x.dtype)
        
        # Assign values from x(k, :) to Dummy where mask is True
        dummy[mask] = x[k, :]

        # Assign Dummy to the appropriate slice of y
        y[:, :, :, k] = dummy

    return y


def Zeppelin(x, g, bval):
    b_delta = 1
    bval = bval.reshape(1,-1).T
    term0 = np.array([np.cos(x[1])*np.sin(x[0]), np.sin(x[0])*np.sin(x[1]), np.cos(x[0])])
    term1 = np.matlib.repmat(term0, g.shape[0], 1)
    dot_prod = (g * term1).sum(axis=1, keepdims=True)

    S = x[4] * np.exp(b_delta * (1/3) * bval * (x[2]-x[3]*x[2]) - bval * (1/3) * (x[3]*x[2]+2*x[2]) - bval * b_delta * dot_prod**2 * (x[2]-x[3]*x[2]))
    return S


# find the b0 image
def find_index(bval):
    min_index_bval = np.argmin(bval)
    return min_index_bval



# root mean squared error
def rmse(array, reference_value):
    differences = array - reference_value             
    #squared_differences = [diff ** 2 for diff in differences] 
    squared_differences = differences ** 2
    mean_squared_error = np.mean(squared_differences)  
    root_mean_squared_error = math.sqrt(mean_squared_error) 
    return root_mean_squared_error


# Computes bvals and bvecs for a given b (b-matrix)
def bval_bvec_from_b_Matrix (b):
    bvals = np.sum(b[:,[0 ,3, 5]],axis=1)
    BSign = np.sign(np.sign(b[:, :3]) + 0.0001)   # Adding 0.0001 avoids getting zeros here
    grad = BSign * np.sqrt(b[:,[0 ,3, 5]])
    grad_n = np.sqrt(np.sum(grad*grad,axis=1))
    grad /= np.column_stack ((grad_n, grad_n, grad_n))
    grad[np.isnan(grad)] = 0
    return bvals,grad
