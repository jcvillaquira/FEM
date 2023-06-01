import numpy as np

import quadrature
from mesh import read_msh

def get_basis_coefficients(n_coor):
    E=np.array([n_coor[:,0]**2,n_coor[:,0]*n_coor[:,1],n_coor[:,1]**2,n_coor[:,0],n_coor[:,1],np.ones(n_coor.shape[0])]).T
    return np.linalg.inv(E)

def get_elemental_matrix(basis_coeff):
    A=np.zeros((6,6))

