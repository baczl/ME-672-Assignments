# Needed for matrix algebra
import numpy as np

# Needed for quadrature integration
from scipy import integrate
# Needed for visualization
import matplotlib.pyplot as plt
# Needed for meshing
from matplotlib.path import Path
from scipy.spatial import Delaunay


# Define heat source function
A = 100
alpha = 100
heat_source_center = (0.5, 0.5)

# OLD FUNCTION
# def f(x,y):
#     return A*np.exp(-alpha*((x-heat_source_center[0])**2 + (y-heat_source_center[1])**2))

# xt = lambda t: (1 - np.cos(t)) / 2

# yt = lambda t: 1 * (
#     0.2969 * np.sqrt(xt(t))
#     - 0.1260 * xt(t)
#     - 0.3516 * xt(t)**2
#     + 0.2843 * xt(t)**3
#     - 0.1036 * xt(t)**4
# )

# x_upper = lambda t: xt(t)
# y_upper = lambda t: yt(t)

# x_lower = lambda t: xt(t)
# y_lower = lambda t: -yt(t)

# Define rectangle dimensions. The shape will always have centroid (0,0).

W = 4
H = 4

corners = np.array([[-W/2,-H/2],[W/2, -H/2],[W/2,H/2],[-W/2,H/2]])



def generate_boundary_nodes(corners, N_BOUNDARY = 4):
    # If there are four nodes, just put them at the corners
    if N_BOUNDARY == 4:
        boundary_nodes = corners
    else:
        # Calculate total perimeter - equal to 4 times the x and y of the top corner
        perimeter = corners[2][0]*4+corners[2][1]*4
        
        # Otherwise, generate boundary nodes
        boundary_nodes = np.zeros((N_BOUNDARY, 2))
        for i in range(0,N_BOUNDARY):
            s = i/N_BOUNDARY*perimeter
            n_segment = 0
            segment_length = 0
            for j in range(4) :
                segment_length = np.linalg.norm(corners[j]-corners[j+1])
                if s < segment_length:
                    n_segment = j
                    break
                else:
                    s -= segment_length
            
            
    

    
    return boundary_nodes

def generate_interior_nodes(boundary_nodes, N_INTERIOR):

    # generate interior nodes
    boundary = Path(boundary_nodes)
    interior_nodes = np.zeros((N_INTERIOR, 2))
    left_node = boundary_nodes[np.argmin(boundary_nodes[:, 0])]
    right_node = boundary_nodes[np.argmax(boundary_nodes[:, 0])]

    for i in range(N_INTERIOR):
        while True:
            x_i = np.random.uniform(left_node[0], right_node[0])
            y_i = np.random.uniform(-1, 1)

            if boundary.contains_point((x_i, y_i)):
                interior_nodes[i] = [x_i, y_i]
                break
    return interior_nodes

def generate_mesh(boundary_nodes, interior_nodes):
    nodes = np.vstack((boundary_nodes, interior_nodes))
    tri = Delaunay(nodes)

    plt.triplot(nodes[:,0], nodes[:,1], tri.simplices)
    return tri, nodes

def assemble_matrix(tri):
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


# def assemble_vector(tri, nodes):


#     b = np.zeros((len(nodes), 1))
#     for triangle in tri.simplices:
#         # Create variables to change depending on triangle dimensions
#         zet = 1/3
#         eps = 1/3

#         # Get the coordinates of the triangle vertices
#         points = tri.points[triangle]
#         # Compute the area of the triangle
#         v1 = points[1] - points[0]
#         v2 = points[2] - points[0]

#         area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

#         # Compute the gradients of the shape functions

#         grads = np.zeros((3, 2))
#         for i in range(3):
#             p1 = points[i]
#             p2 = points[(i+1)%3]
#             p3 = points[(i+2)%3]
#             grads[i] = np.array([p2[1] - p3[1], p3[0] - p2[0]]) / (2 * area)

#         def x_standardized(zet,eps):
#             return p1[0] + (p2[0] - p1[0]) * zet + (p3[0] - p1[0]) * eps
        
#         def y_standardized(zet,eps):
#             return p1[1] + (p2[1] - p1[1]) * zet + (p3[1] - p1[1]) * eps

#         def f_standardized(x_standardized,y_standardized):
#             return f(x_standardized,y_standardized)

#         for i in range(3):
#             # Get the index f of the node in the global system
#             j = triangle[i]
#             # Compute the integral of f * N_i over the triangle in pieces
#             integral_justf = (1-grads[i][0]*nodes[j][0]-grads[i][1]*nodes[j][1]) * sp.integrate(f_standardized, (eps, bounds[0,0], bounds[0,1]), (zet, bounds[1,0], bounds[1,1]))
#             integral_x = (grads[i][0]*nodes[j][0]) * sp.integrate(f_standardized, (eps, bounds[0,0], bounds[0,1]), (zet, bounds[1,0], bounds[1,1]))
#             integral_y = (grads[i][1]*nodes[j][1]) * sp.integrate(f_standardized, (eps, bounds[0,0], bounds[0,1]), (zet, bounds[1,0], bounds[1,1]))
#             # b[j] += integral_justf + integral_x + integral_y
#             print(integral_justf)
#     return b

def solve_system(K, f):
    # Solve the system of equations Ku = f
    u = np.linalg.solve(K, f)
    return u

boundary_nodes = generate_boundary_nodes(x_upper, y_upper, x_lower, y_lower, N_BOUNDARY=7)

interior_nodes = generate_interior_nodes(boundary_nodes, N_INTERIOR=20)

tri, nodes = generate_mesh(boundary_nodes, interior_nodes)

K = assemble_matrix(tri)

# b = assemble_vector(tri, nodes)


plt.plot(x_upper(np.linspace(0, np.pi, 100)), y_upper(np.linspace(0, np.pi, 100)), color='blue')
plt.plot(x_lower(np.linspace(0, np.pi, 100)), y_lower(np.linspace(0, np.pi, 100)), color='blue')

plt.axis('equal')
plt.legend()
plt.show()