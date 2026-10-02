import numpy as np
from scipy.spatial import Delaunay
from scipy.integrate import dblquad

G = lambda xi, eta: np.array([[0.25*eta - 0.25, 0.25*xi - 0.25],
                              [0.25 - 0.25*eta, -0.25*xi - 0.25],
                                [0.25*eta + 0.25, 0.25*xi + 0.25],
                                [-0.25*eta - 0.25, 0.25 - 0.25*xi]])

N = lambda xi, eta: np.array([1/4 * (1-xi)*(1-eta), 
                            1/4 * (1+xi)*(1-eta), 
                            1/4 * (1+xi)*(1+eta), 
                            1/4 * (1-xi)*(1+eta)])

def assemble_matrix(tri, D):
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
        K_local = area * (G @ D @ G.T)
        for i in range(3):
            for j in range(3):
                K[triangle[i], triangle[j]] += K_local[i, j]
    
    return K

def assemble_vector(tri, f) :
    N_NODES = len(tri.points)
    b = np.zeros(N_NODES)

    for triangle in tri.simplices:
        
        # Get the coordinates of the triangle vertices
        points = tri.points[triangle]

        # Compute the gradients of the shape functions

        # Calculate the jacobian
        J = np.array([
    [points[1,0] - points[0,0], points[2,0] - points[0,0]],
    [points[1,1] - points[0,1], points[2,1] - points[0,1]]
])
        # Make shape functions
        N1 = lambda xi, eta: 1-xi-eta
        N2 = lambda xi, eta: xi
        N3 = lambda xi, eta: eta

        N = [N1, N2, N3]

        # Make map functions
        x_map = lambda xi, eta: N1(xi,eta)*points[0][0]+ N2(xi,eta)*points[1][0]+ N3(xi,eta)*points[2][0]
        y_map = lambda xi, eta: N1(xi,eta)*points[0][1]+ N2(xi,eta)*points[1][1]+ N3(xi,eta)*points[2][1]

        scale = abs(np.linalg.det(J))
        
        for i in range(3):
            # Get the index f of the node in the global system
            j = triangle[i]
            b[j] += scale*dblquad(lambda eta, xi: N[i](xi,eta)*f(x_map(xi,eta), y_map(xi,eta)), 0, 1, lambda xi: 0, lambda xi: 1-xi)[0]
    return b

def neuman_bc(b, mesh, boundary_conditions, q0):
    N_BOUNDARY = len(mesh.boundary_nodes)
    # Check which side indices are neuman condition
    indices = np.array([i for i, val in enumerate(boundary_conditions) if val == 'n'])
    for i in range(0,N_BOUNDARY):
        # Get a pair of points at indices i and j
        j = (i+1) % N_BOUNDARY
        # Make sure they are both neuman condition
        if np.any(indices == mesh.segment_n[i]) and np.any(indices == mesh.segment_n[j]):
            n = [mesh.boundary_nodes[i], mesh.boundary_nodes[j]]
            edge = n[1]-n[0]
            L = np.linalg.norm(edge)

            # Integrate q0 quich is a constant
            b[mesh.boundary_ids[i]] += q0 * L / 2
            b[mesh.boundary_ids[j]] += q0 * L / 2

    return b

def dirichlet_bc(K,b, mesh, boundary_conditions, dirichlet_values):
    # Find edge indices that are dirichlet
    indices = np.array([i for i, val in enumerate(boundary_conditions) if val == 'd'])
    # If there are not zero dirichlet edges, modify K and b accordingly
    if not (len(indices) == 0):
        # Find out which nodes are dirichlet
        dirichlet_nodes = np.array([
        j
        for j in range(0,len(mesh.boundary_nodes))
        if boundary_conditions[mesh.segment_n[j]] == 'd'
    ])

        # Find all unknown indices
        unknown_ids = np.setdiff1d(np.arange(len(b)), mesh.boundary_ids[dirichlet_nodes])
        # Loop over the vector and over each dirichlet column of K subracting all entries times the dirichlet conditions
        for i in unknown_ids:
            for j in dirichlet_nodes: 
                #Subtract the average known value from the right side if its a corner dirichlet node
                # if np.any(corner_nodes == j):
                #     corner_index = np.where(corner_nodes == j)[0][0]
                #     b[i] -= K[i,mesh.boundary_ids[j]]*1/2*(dirichlet_values[int(adjacent_indices[corner_index][0])] + dirichlet_values[int(adjacent_indices[corner_index][1])])

                # else:
                #     # Otherwise just subtract the known value from the right side
                b[i] -= K[i,mesh.boundary_ids[j]]*dirichlet_values[int(mesh.segment_n[j])]

        # If boundary value is known (dirichlet), remove columns and rows that have it.

        b = np.delete(b, mesh.boundary_ids[dirichlet_nodes])
        K = np.delete(K, mesh.boundary_ids[dirichlet_nodes], axis=0)
        K = np.delete(K, mesh.boundary_ids[dirichlet_nodes], axis=1)
    return K, b

def gen(mesh, f, q0, boundary_conditions, dirichlet_values, D):
    if mesh.type == 'tri' :
        K = assemble_matrix(mesh.tri, D)
        b = assemble_vector(mesh.tri, f)
        b = neuman_bc(b, mesh, boundary_conditions, q0)
        (K, b) = dirichlet_bc(K,b, mesh, boundary_conditions, dirichlet_values)
    elif mesh.type == 'quad':
        K = assemble_matrix_quad(mesh, D)
        b = assemble_vector_quad(mesh,f)
        b = neuman_bc(b, mesh, boundary_conditions, q0)
        (K, b) = dirichlet_bc(K,b, mesh, boundary_conditions, dirichlet_values)
    return K, b

def assemble_matrix_quad(mesh, D):
        quad = mesh.rect
    # Find how many total nodes there are
        N_NODES = len(quad.points)
        K = np.zeros((N_NODES, N_NODES))
        
        for q in quad.simplices:
            # Get the coordinates of the quadrilateral vertices -- must be in counter clockwise order

            p = np.array([quad.points[i] for i in q])         
            
            # Define Jacobian and determinant as a function of xi and eta

            J = lambda xi, eta: p.T @ G(xi,eta)

            detJ = lambda xi, eta: np.abs(np.linalg.det(J(xi, eta)))

            # B is Gradient scale factor for use in integration

            B = lambda xi, eta: G(xi, eta) @ np.linalg.inv(J(xi,eta))
            # Make local 4x4 matrix for all integrals in quad
            K_local = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    # Integrate everything with scaled gradient and dA = detJ dxi,deta
                    K_local[i,j] = dblquad(
                        lambda eta, xi:
                            B(xi,eta)[i] @ D @ B(xi,eta)[j].T
                            * detJ(xi,eta),
                        -1, 1,
                        lambda xi: -1,
                        lambda xi: 1
                    )[0]
            # Add integrations at their positions in the global matrix
            for i in range(4):
                for j in range(4):
                    K[q[i], q[j]] += K_local[i, j]
        
        return K

def assemble_vector_quad(mesh, f) :
    quad = mesh.rect
    N_NODES = len(quad.points)
    b = np.zeros(N_NODES)

    for q in quad.simplices:
        
        # Get the coordinates of the vertices
        p = np.array([quad.points[i] for i in q])         

        J = lambda xi, eta: p.T @ G(xi,eta)

        
        # Make map functions

        x_map = lambda xi,eta: N(xi, eta) @ p[:, 0]
        y_map = lambda xi,eta: N(xi, eta) @ p[:, 1]

        detJ = lambda xi, eta: abs(np.linalg.det(J(xi, eta)))
        
        for i in range(4):
            # Get the index f of the node in the global system
            j = q[i]
            b[j] += dblquad(lambda eta, xi: N(xi,eta)[i]*f(x_map(xi,eta), y_map(xi,eta)) * detJ(xi, eta), -1, 1, lambda xi: -1, lambda xi: 1)[0]
    return b

def gen_quad(mesh, f, q0, boundary_conditions, dirichlet_values, D):
    K = assemble_matrix_quad(mesh, D)
    b = assemble_vector_quad(mesh,f)
    b = neuman_bc(b, mesh, boundary_conditions, q0)
    (K, b) = dirichlet_bc(K,b, mesh, boundary_conditions, dirichlet_values)
    return (K,b)