import numpy as np

import quadrature
from mesh import read_msh

node_coordinates,connection_table=read_msh('L2.msh')
print(node_coordinates[0])

def get_basis_coefficients(n_coor):
    E=np.array([n_coor[:,0]**2,n_coor[:,0]*n_coor[:,1],n_coor[:,1]**2,n_coor[:,0],n_coor[:,1],np.ones(n_coor.shape[0])]).T
    return np.linalg.inv(E)
