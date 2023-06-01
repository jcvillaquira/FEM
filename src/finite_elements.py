import numpy as np

from quadrature import triangle_cuadrature
from mesh import read_msh

node_coordinates,connection_table=read_msh('L2.msh')
print(node_coordinates[0])

class FEM_Equation_Solver():
    def __init__(self,node_coordinates,connection_table):
        self.node_coordinates=node_coordinates
        self.connection_table=connection_table
        self.stiffness_matrix=np.zeros(self.node_coordinates.shape[0])

    def c(self,x,y):
        return 0.0
    
    def f(self,x,y):
        return x+y
        

    def get_basis_coefficients(self,n_coor):
        E=np.array([n_coor[:,0]**2,n_coor[:,0]*n_coor[:,1],n_coor[:,1]**2,n_coor[:,0],n_coor[:,1],np.ones(n_coor.shape[0])]).T
        return np.linalg.inv(E).T

    def get_elemental_matrix(self,basis_coeff,vertex_coordinates):
        A=np.zeros((6,6))
        for i in range(6):
            for j in range(i,6):
                def f_tot(x,y):
                    t1=(2*basis_coeff[i][0]*x+basis_coeff[i][1]*y+basis_coeff[i][3])
                    t2=(2*basis_coeff[j][0]*x+basis_coeff[j][1]*y+basis_coeff[j][3])
                    t3=(basis_coeff[i][1]*x+2*basis_coeff[i][2]*y+basis_coeff[i][4])
                    t4=(basis_coeff[j][1]*x+2*basis_coeff[j][2]*y+basis_coeff[j][4])
                    t5=(basis_coeff[i][0]*x**2+
                        basis_coeff[i][1]*x*y+
                        basis_coeff[i][2]*y**2+
                        basis_coeff[i][3]*x+
                        basis_coeff[i][4]*y+
                        basis_coeff[i][5])
                    
                    t6=(basis_coeff[j][0]*x**2+
                        basis_coeff[j][1]*x*y+
                        basis_coeff[j][2]*y**2+
                        basis_coeff[j][3]*x+
                        basis_coeff[j][4]*y+
                        basis_coeff[j][5])
                    return t1*t2+t3*t4+self.c(x,y)*t5*t6
            A[i,j]=triangle_cuadrature(f_tot,vertex_coordinates)
        return A
    