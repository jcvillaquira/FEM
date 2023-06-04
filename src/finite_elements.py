import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix,coo_array

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

import matplotlib.tri as mtri 
import scipy.spatial


from .quadrature import triangle_cuadrature

class FEM_Equation_Solver():
    def __init__(self,node_coordinates,connection_table,dirichlet_nodes,c_function,f_function):
        self.node_coordinates=node_coordinates
        self.connection_table=connection_table
        self.dirichlet_nodes=dirichlet_nodes
        self.c=c_function
        self.f=f_function
        basis_coeff=[]
        for i in range(len(self.connection_table)):
            basis_coeff.append(self.get_basis_coefficients(self.node_coordinates[self.connection_table[i]]))
        self.basis_coeff=basis_coeff
        self.stiffness_matrix=self.assembly_stiffness_matrix()
        self.A=self.assembly_A()
        self.stiffness_I=sp.sparse.linalg.splu(self.stiffness_matrix)

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
    
    def get_elemental_A(self,basis_coeff,element_nodes,vertex_coordinates):
        indx=[]
        indy=[]
        data=[]
        for i in range(len(element_nodes)):
            for j in range(i,len(element_nodes)):
                
                def f_tot(x,y):
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
                    return t5*t6
                """
                def f_tot(x,y):
                    t3=(basis_coeff[i][0]*x+basis_coeff[i][1]*y+basis_coeff[i][2])
                    t4=(basis_coeff[j][0]*x+basis_coeff[j][1]*y+basis_coeff[j][2])
                    return t3*t4
                """
                indx.append(element_nodes[i])
                indy.append(element_nodes[j])
                data.append(triangle_cuadrature(f_tot,vertex_coordinates))
        return indx,indy,data
    
    def assembly_A(self):
        indx=[]
        indy=[]
        data=[]
        for i in range(len(self.connection_table)):
            node_coor=self.node_coordinates[self.connection_table[i]]
            indxe,indye,datae=self.get_elemental_A(self.basis_coeff[i],self.connection_table[i],node_coor[:3])
            indx+=indxe
            indy+=indye
            data+=datae
        indx=np.array(indx)
        indy=np.array(indy)
        data=np.array(data)

        mat=coo_matrix((data, (indx, indy)), shape=(self.node_coordinates.shape[0], self.node_coordinates.shape[0])).tocsr()
        mat[indy,indx]=mat[indx,indy]

        #for node in self.dirichlet_nodes:
            #mat[node,:]=0.0
            #mat[node,node]=1.0

        return mat

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
    
    def get_elemental_load_vector(self,basis_coeff,element_nodes,vertex_coordinates,f_i):
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
                return t1*f_i(x,y)
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
    
    def assembly_load_vector(self, f_i,force_dirichlet=True):
        b=np.zeros(self.node_coordinates.shape[0])
        for i, element in enumerate( self.connection_table ):
            node_coor=self.node_coordinates[element]
            indbe,databe=self.get_elemental_load_vector(self.basis_coeff[i],
                                                        self.connection_table[i],
                                                        node_coor[:3],f_i)
            for i in range(len(indbe)):
                b[indbe[i]]+=databe[i]

        for node in self.dirichlet_nodes:
            x,y=self.node_coordinates[node]
            b[node]=self.g(x,y)

        return b
    
    def set_vector_dirichlet(self,b):
        bnew=b
        for node in self.dirichlet_nodes:
            x,y=self.node_coordinates[node]
            bnew[node]=self.g(x,y)
        return bnew


    def assembly_stiffness_matrix(self):
        indx=[]
        indy=[]
        data=[]
        for i in range(len(self.connection_table)):
            node_coor=self.node_coordinates[self.connection_table[i]]
            indxe,indye,datae=self.get_elemental_matrix(self.basis_coeff[i],self.connection_table[i],node_coor[:3])
            indx+=indxe
            indy+=indye
            data+=datae
        indx=np.array(indx)
        indy=np.array(indy)
        data=np.array(data)

        mat=coo_matrix((data, (indx, indy)), shape=(self.node_coordinates.shape[0], self.node_coordinates.shape[0])).tocsr()
        mat[indy,indx]=mat[indx,indy]

        for node in self.dirichlet_nodes:
            mat[node,:]=0.0
            mat[node,node]=1.0

        return mat
    
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

class Heat_Equation_Solver():
    def __init__(self,node_coordinates,connection_table,dirichlet_nodes,u0,dt,T_fin, f_function):
        self.node_coordinates=node_coordinates
        self.connection_table=connection_table
        self.dirichlet_nodes=dirichlet_nodes
        self.f_function=f_function
        self.dt=dt
        self.T_fin=T_fin
        self.diffusion_const=0.1
        def c(x,y):
            return 1.0/(self.dt*self.diffusion_const)
        def f_init(x,y):
            return self.f_function(0.0,x,y)
        self.fem_solver=FEM_Equation_Solver(node_coordinates,connection_table,dirichlet_nodes,c,f_init)
        self.u0_V=self.u0_project(u0)
    
    def u0_project(self,u0):
        u0_V=np.zeros((self.node_coordinates.shape[0]))
        for i in range(len(self.node_coordinates)):
            x,y=self.node_coordinates[i]
            u0_V[i]=u0(x,y)
        return u0_V

    def u0_project_old(self,u0):
        uv0 = self.fem_solver.assembly_load_vector(u0,force_dirichlet=True)
        return uv0

    def f_project(self,t):
        f_p=np.zeros((self.node_coordinates.shape[0]))
        for i in range(len(self.node_coordinates)):
            x,y=self.node_coordinates[i]
            f_p[i]=self.f_function(t,x,y)
        return f_p
    
    
    def solve(self):
        solution=[self.u0_V]
        t=0.0
        i=1
        def f_i(x,y):
            return self.f_function(0.0,x,y)
        f_pt=self.fem_solver.assembly_load_vector(f_i,force_dirichlet=True)
        while t<=self.T_fin:
            t1=self.fem_solver.A.dot(solution[-1])
            vecnew=self.fem_solver.set_vector_dirichlet((f_pt+(t1/self.dt))/self.diffusion_const)
            unew=self.fem_solver.stiffness_I.solve(vecnew)
            solution.append(unew)
            #print(np.max(unew))
            t+=self.dt 
            i+=1
        return np.array(solution)
    
    def plot_solution(self, sol):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Xs = self.node_coordinates[:,0]
        Ys = self.node_coordinates[:,1]

        pts = np.vstack([Xs, Ys]).T
        tess = scipy.spatial.Delaunay(pts)

        xx = tess.points[:, 0]
        yy = tess.points[:, 1]
        tri = tess.vertices 
        tri_new=[]
        for trss in tri:
            ps=[]
            for p in trss:
                if xx[p]==1.0 and yy[p]>1.0:
                    break
                else:
                    ps.append(p)
            if len(ps)==3:
                tri_new.append(ps)
        triDat = mtri.Triangulation(x=pts[:, 0], y=pts[:, 1], triangles=tri_new)

        surf = ax.plot_trisurf(triDat, sol[0], cmap=cm.jet, linewidth=0)
        fig.colorbar(surf)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlim(0,2)
        ax.set_ylim(0,2)
        ax.set_zlim(np.min(sol[0]),np.max(sol[0])*1.5)

    

        fig.tight_layout()

        fig.subplots_adjust(bottom=0.25)

        axfreq = fig.add_axes([0.25, 0.1, 0.5, 0.03])
        freq_slider = Slider(
            ax=axfreq,
            label='t',
            valmin=0.0,
            valmax=self.T_fin,
            valinit=0.0,
            valstep=self.dt
        )
        times=np.arange(0,self.T_fin,self.dt)
        vmin=np.min(sol[0])
        vmax=np.max(sol[0])

        def draw(t):  
            ax.cla()
            i=np.argmin(np.abs(times-t))
            #print(i)
            ax.set_xlim(0,2)
            ax.set_ylim(0,2)
            ax.set_zlim(vmin,vmax*1.5)
            ax.plot_trisurf(triDat, sol[i], cmap=cm.jet, linewidth=0,vmin=vmin, vmax=vmax)
        draw(0)
        freq_slider.on_changed(draw)
        return freq_slider






    
