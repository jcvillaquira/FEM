
import os
import numpy as np
from pathlib import Path

path = Path(__file__).parent.absolute()
os.chdir(path)
#points=np.array([[0.5,0.0],[0.5,0.5],[0.0,0.5]])
#points=points.T
#weigths=np.array([1.0/3,1.0/3,1.0/3])

points=np.loadtxt('strang4_x.txt').T
weigths=np.loadtxt('strang4_w.txt')


#def f(x,y):
    #return x+y
#vertex=np.array([[0.0,0.0],[1.0,1.0],[2.0,0.0]])


def triangle_cuadrature(f,vertex):
    A=np.array([vertex[1]-vertex[0],vertex[2]-vertex[0]]).T
    points_x,points_y=np.dot(A,points)+np.tile(vertex[0],(points.shape[1],1)).T
    f_points=f(points_x,points_y)
    return np.abs(np.linalg.det(A))*np.sum(weigths*f_points)/2.0

#print(triangle_cuadrature(f,vertex))