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
# os.chdir('/home/julian/Personal/FEM/')

# %%
from src.mesh import read_msh
from src.quadrature import *
from src.finite_elements import FEM_Equation_Solver

# %%
node_coordinates, connection_table = read_msh('data/L2.msh')
fem_solver = FEM_Equation_Solver(node_coordinates, connection_table)

