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
import os
import numpy as np
import scipy as sp
os.chdir('/home/julian/Personal/FEM/')

# %%
from src.mesh import read_msh
from src.quadrature import *
from src.finite_elements import *

# %%
node_coordinates, connection_table, dirichlet_nodes = read_msh('data/L2.msh')

# %%
x1 = 1.0
y1 = 1.0
x_coordinates = node_coordinates[:, 0]
y_coordinates = node_coordinates[:, 1]
not_boundary = (x_coordinates == x1) & (y_coordinates < y1) & (y_coordinates > 0.0)
not_boundary_idx = set(np.array(range(len(not_boundary)))[not_boundary])
dirichlet_nodes_modified = set(dirichlet_nodes) - not_boundary_idx

# %%
def u0(x,y):
    if (x-0.5)**2 +(y-1.5)**2 < 0.25**2:
        return 1.0
    else:
        return 1.0
def f_function(t,x,y):
    #return 1.0
    if (x-0.5)**2 +(y-1.5)**2 < 0.25**2:
        return 0.0
    else:
        return 0.0

# %%
f_function_vec=np.vectorize(f_function,excluded=[0])
u0_vec=np.vectorize(u0)
dt=0.01
T_fin=1.0
heat_solver=Heat_Equation_Solver(node_coordinates,connection_table,dirichlet_nodes,u0_vec,dt,T_fin,f_function_vec)

# %%
solution=heat_solver.solve()

# %%
heat_solver.fem_solver.plot_solution(solution[1])

# %%
heat_solver.plot_solution(solution)

# %%
np.max(solution[-1])

# %%
mat,b=fem_solver.assembly_stiffness_matrix_and_load_vector()
Zs=sp.sparse.linalg.spsolve(mat,b)

# %%
fem_solver.plot_solution(Zs)

# %%
A=mat.toarray()

# %%
A[9]

# %%
b

# %%
plt.spy(mat)

# %%
np.max(b)

# %%
sp.sparse.linalg.spsolve(A,b)

# %%
np.linalg.solve(A, b)

