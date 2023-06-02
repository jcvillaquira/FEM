import numpy as np
from scipy.sparse import coo_matrix,coo_array

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


from .quadrature import triangle_cuadrature

class FEM_Equation_Solver():
    def __init__(self,node_coordinates,connection_table,dirichlet_nodes):
        self.node_coordinates=node_coordinates
        self.connection_table=connection_table
        self.dirichlet_nodes=dirichlet_nodes

        #self.stiffness_matrix=np.zeros(self.node_coordinates.shape[0])
        #self.stiffness_matrix=coo_matrix(shape=(self.node_coordinates.shape[0], self.node_coordinates.shape[0]))

    def c(self,x,y):
        return 0.0
    
    def f(self,x,y):
        return -26
    
    def g(self,x,y):
        return 0.0
        if np.abs(y)<1e-5:
            return 3*x**2-x
        if np.abs(x-1)<1e-5:
            return 3+10*y**2-1
        if np.abs(y-1)<1e-5:
            return 3*x**2-2*x+11
        if np.abs(x)<1e-5:
            return 10*y**2+y
    
        

    def get_basis_coefficients(self,n_coor):
        """
        n_coor is a 6x2 matrix containing in its columns the coordinates of the six nodes in an elements
        It return the matrix whose columns are the coefficients associated with the each of basis function in the form
        phi=ax^2+bxy+cy^2+dx+ey+f for every node.
        """
        E=np.array([n_coor[:,0]**2,n_coor[:,0]*n_coor[:,1],n_coor[:,1]**2,n_coor[:,0],n_coor[:,1],np.ones(n_coor.shape[0])]).T
        #E=np.array([n_coor[:,0],n_coor[:,1],np.ones(n_coor.shape[0])]).T
        return np.linalg.inv(E).T

    def get_elemental_matrix(self,basis_coeff,element_nodes,vertex_coordinates):
        """
        basis_coeff: is the matrix containing the coefficients associated to the node basis functions in its columns
        vertex_coordinates is a 3x2 matrix containing the coordinates of the triangle in its rows

        """
        indx=[]
        indy=[]
        data=[]
        for i in range(len(element_nodes)):
            for j in range(i,len(element_nodes)):
                
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
                """
                def f_tot(x,y):
                    t1=(basis_coeff[i][0])*(basis_coeff[j][0])
                    t2=(basis_coeff[i][1])*(basis_coeff[j][1])
                    t3=(basis_coeff[i][0]*x+basis_coeff[i][1]*y+basis_coeff[i][2])
                    t4=(basis_coeff[j][0]*x+basis_coeff[j][1]*y+basis_coeff[j][2])
                    return t1+t2+(self.c(x,y)*t3*t4)
                """
                indx.append(element_nodes[i])
                indy.append(element_nodes[j])
                data.append(triangle_cuadrature(f_tot,vertex_coordinates))
        return indx,indy,data
    
    def get_elemental_load_vector(self,basis_coeff,element_nodes,vertex_coordinates):
        ind=[]
        data=[]
        for i in range(len(element_nodes)):
            
            def f_b(x,y):
                t1=(basis_coeff[i][0]*x**2+
                        basis_coeff[i][1]*x*y+
                        basis_coeff[i][2]*y**2+
                        basis_coeff[i][3]*x+
                        basis_coeff[i][4]*y+
                        basis_coeff[i][5])
                return t1*self.f(x,y)
            """
            def f_b(x,y):
                t1=(basis_coeff[i][0]*x+
                    basis_coeff[i][1]*y+
                    basis_coeff[i][2])
                return t1*self.f(x,y)
            """
            ind.append(element_nodes[i])
            data.append(triangle_cuadrature(f_b,vertex_coordinates[:3]))
        return ind,data



    def assembly_stiffness_matrix_and_load_vector(self):
        indx=[]
        indy=[]
        #indb=[]
        data=[]
        #datab=[]
        b=np.zeros(self.node_coordinates.shape[0])
        for i in range(len(self.connection_table)):
            node_coor=self.node_coordinates[self.connection_table[i]]
            #print("Elemento ",i ,self.connection_table[i])
            basis_coeff=self.get_basis_coefficients(node_coor)
            #print(basis_coeff)
            indxe,indye,datae=self.get_elemental_matrix(basis_coeff,self.connection_table[i],node_coor[:3])
            #print(indxe,indye,datae)
            indbe,databe=self.get_elemental_load_vector(basis_coeff,self.connection_table[i],node_coor[:3])
            for i in range(len(indbe)):
                b[indbe[i]]+=databe[i]
            indx+=indxe
            indy+=indye
            #indb+=indbe
            data+=datae
            #datab+=databe
        indx=np.array(indx)
        indy=np.array(indy)
        data=np.array(data)

        mat=coo_matrix((data, (indx, indy)), shape=(self.node_coordinates.shape[0], self.node_coordinates.shape[0])).tocsr()
        mat[indy,indx]=mat[indx,indy]

        for node in self.dirichlet_nodes:
            mat[node,:]=0.0
            mat[node,node]=1.0
            x,y=self.node_coordinates[node]
            b[node]=self.g(x,y)

        return mat,b
    
    def plot_solution(self,Zs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Xs=self.node_coordinates[:,0]
        Ys=self.node_coordinates[:,1]

        surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
        fig.colorbar(surf)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        fig.tight_layout()

        plt.show() # or:
    
