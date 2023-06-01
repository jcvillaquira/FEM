# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Ecopetrol
#     language: python
#     name: python3
# ---

# %%
%load_ext autoreload
%autoreload 2

# %%
from src.mesh import read_msh
from src.quadrature import *
import numpy as np

# %%

node_coordinates,connection_table=read_msh('L2.msh')
print(node_coordinates[0])

def get_basis_coefficients(n_coor):
    E=np.array([n_coor[:,0]**2,n_coor[:,0]*n_coor[:,1],n_coor[:,1]**2,n_coor[:,0],n_coor[:,1],np.ones(n_coor.shape[0])]).T
    return np.linalg.inv(E)

def get_elemental_matrix(basis_coeff):
    A=np.zeros((6,6))


