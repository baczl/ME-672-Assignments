import numpy as np
from scipy.spatial import Delaunay
from scipy.integrate import quad


def assemble_matrix(tri):
    # Find how many total nodes there are
    N_NODES = len(tri.points)
    K = np.zeros((N_NODES, N_NODES))
    for triangle in tri.simplices:
        # Get the coordinates of the triangle vertices
        p1 = tri.points[triangle[0]]
        p2 = tri.points[triangle[1]]
        p3 = tri.points[triangle[2]]

        # Compute the area of the triangle
        v1 = p2 - p1
        v2 = p3 - p1

        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

        # Compute the gradients of the shape functions
        grad_N1 = np.array([p2[1] - p3[1], p3[0] - p2[0]]) / (2 * area)
        grad_N2 = np.array([p3[1] - p1[1], p1[0] - p3[0]]) / (2 * area)
        grad_N3 = np.array([p1[1] - p2[1], p2[0] - p1[0]]) / (2 * area)

        # Assemble the local stiffness matrix
        G = np.array([grad_N1, grad_N2, grad_N3])
        K_local = area * (G @ G.T)
        for i in range(3):
            for j in range(3):
                K[triangle[i], triangle[j]] += K_local[i, j]
    
    return K

def assemble_vector(tri) :
    N_NODES = len(tri.points)
    nodes = tri.points
    b = np.zeros((N_NODES, 1))
    for triangle in tri.simplices:
        # Create variables to change depending on triangle dimensions
        zet = 1/3
        eps = 1/3

        # Get the coordinates of the triangle vertices
        points = tri.points[triangle]
        # Compute the area of the triangle
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

        # Compute the gradients of the shape functions

        grads = np.zeros((3, 2))
        for i in range(3):
            p1 = points[i]
            p2 = points[(i+1)%3]
            p3 = points[(i+2)%3]
            grads[i] = np.array([p2[1] - p3[1], p3[0] - p2[0]]) / (2 * area)

        def x_standardized(zet,eps):
            return p1[0] + (p2[0] - p1[0]) * zet + (p3[0] - p1[0]) * eps
        
        def y_standardized(zet,eps):
            return p1[1] + (p2[1] - p1[1]) * zet + (p3[1] - p1[1]) * eps

        def f_standardized(x_standardized,y_standardized):
            return f(x_standardized,y_standardized)

        # for i in range(3):
        #     # Get the index f of the node in the global system
        #     j = triangle[i]
        #     # Compute the integral of f * N_i over the triangle in pieces
        #     integral_justf = (1-grads[i][0]*nodes[j][0]-grads[i][1]*nodes[j][1]) * quad(f_standardized, (eps, bounds[0,0], bounds[0,1]), (zet, bounds[1,0], bounds[1,1]))
        #     integral_x = (grads[i][0]*nodes[j][0]) * quad(f_standardized, (eps, bounds[0,0], bounds[0,1]), (zet, bounds[1,0], bounds[1,1]))
        #     integral_y = (grads[i][1]*nodes[j][1]) * quad(f_standardized, (eps, bounds[0,0], bounds[0,1]), (zet, bounds[1,0], bounds[1,1]))
        #     # b[j] += integral_justf + integral_x + integral_y
        #     print(integral_justf)
    return b

def gen(tri) :
    K = assemble_matrix(tri)
    b = assemble_vector(tri)

    return K, b