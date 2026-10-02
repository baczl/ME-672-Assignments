# Needed for matrix algebra
import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay

from dataclasses import dataclass

@dataclass
class Mesh:
    boundary_nodes: np.ndarray
    segment_n: np.ndarray
    nodes: np.ndarray
    corner_nodes: np.ndarray
    adjacent_indices: np.ndarray
    type: str
    interior_nodes: np.ndarray = None
    tri: Delaunay = None
    rect: Rectangle = None
    boundary_ids: np.ndarray = None

@dataclass
class Rectangle:
    simplices: np.ndarray
    points: np.ndarray   

def gen_boundary_nodes(corners, N_BOUNDARY=4):
    segment_n = [0] * N_BOUNDARY
    # If there are four nodes, just put them at the corners
    if N_BOUNDARY == 4:
        boundary_nodes = corners
        segment_n = [0, 1, 2, 3]
    else:
        # Calculate total perimeter - equal to 4 times the x and y of the top corner
        perimeter = 2*np.linalg.norm(corners[1]-corners[0])+2*np.linalg.norm(corners[2]-corners[1])
        # Generate boundary nodes
        boundary_nodes = np.zeros((N_BOUNDARY, 2))
        # Loop through all boundary nodes
        for i in range(0,N_BOUNDARY):
            # Define total length starting from bottom left corner
            s = (i)/(N_BOUNDARY)*perimeter
            n_segment = 0
            segment_length = np.linalg.norm(corners[(n_segment + 1) % 4] - corners[n_segment])
            while s > segment_length :
                n_segment += 1
                s -= segment_length
                segment_length = np.linalg.norm(corners[(n_segment + 1) % 4] - corners[n_segment])
            segment_n[i] = n_segment
            boundary_nodes[i] = (corners[(n_segment + 1) % 4] - corners[n_segment]) * s/segment_length + corners[n_segment]
    
    return boundary_nodes, segment_n

def rect_boundary_nodes(corners, N_X, N_Y):
    delta_x = (corners[2][0] - corners[0][0])/(N_X-1)
    delta_y = (corners[2][1] - corners[0][1])/(N_Y-1)

    N_BOUNDARY = 2*N_X+2*N_Y - 4

    segment_n = [0] * N_BOUNDARY

    boundary_nodes = np.zeros((N_BOUNDARY, 2))
    boundary_ids = np.zeros(N_BOUNDARY, dtype=int)

    def grid_number(i,j):
        return (i)*(N_Y) + (j)
    
    n = 0
    for i in range(0, N_X):
        boundary_nodes[n] = [i*delta_x+corners[0][0], corners[0][1]]
        boundary_ids[n] = grid_number(i, 0)
        segment_n[n] = 0
        n+=1
    for i in range(1, N_Y-1):
        boundary_nodes[n] = [corners[1][0], i*delta_y+corners[1][1]]
        boundary_ids[n] = grid_number(N_X-1, i)
        segment_n[n] = 1
        n+=1
    for i in range(N_X-1, -1, -1):
        boundary_nodes[n] = [i*delta_x+corners[0][0], corners[2][1]]
        boundary_ids[n] = grid_number(i, N_Y-1)
        segment_n[n] = 2
        n+=1
    for i in range(N_Y-2, 0, -1):
        boundary_nodes[n] = [corners[3][0], i*delta_y+corners[0][1]]
        boundary_ids[n] = grid_number(0, i)
        segment_n[n] = 3
        n+=1    
    return boundary_nodes, boundary_ids, segment_n

def gen_interior_nodes(corners, N_INTERIOR = 1):
    if N_INTERIOR == 1:
        interior_nodes = corners[2]-0.5*(corners[2]-corners[0])
    else:
        # Otherwise just generate randoms inside boundary
        rng = np.random.default_rng()
    
        interior_nodes = np.column_stack((  

        rng.uniform(low=corners[0,0], high=corners[2,0], size=N_INTERIOR),
        rng.uniform(low=corners[0,1], high=corners[2,1], size=N_INTERIOR)))

    return interior_nodes

def rectangle_mesh(corners, N_X, N_Y):
    # Just fill the rectangular boundary with evenly spaced rectangles
    # Start at bottom left corner make first rectangle
    simplices = np.zeros(((N_X-1)* (N_Y-1), 4), dtype=int)
    starting_corner = corners[0]
    scale = np.array([[1/(N_X-1), 0], [0, 1/(N_Y-1)]])
    delta = corners[2]-corners[0]
    N_BOUNDARY = N_X*2+N_Y*2-4
    
    (boundary_nodes, boundary_ids, segment_n) = rect_boundary_nodes(corners, N_X,N_Y)
    (corner_nodes, adjacent_indices) = get_corner_nodes(corners, boundary_nodes)
    
    points = np.zeros((N_X * N_Y, 2))
    # points[0:N_BOUNDARY] = boundary_nodes
    def grid_number(i,j):
        return (i)*(N_Y) + (j)

    def element_number(i,j):
        return i*(N_Y-1) + j
    for i in range(0, N_X):
        for j in range(0, N_Y):
            points[grid_number(i,j)] = starting_corner + (scale @ delta) * ([i,j])

    for i in range(0, N_X-1):
        for j in range(0, N_Y-1):
            simplices[element_number(i, j)] = [
                grid_number(i, j),
                grid_number(i + 1, j),
                grid_number(i + 1, j + 1),
                grid_number(i, j + 1)
            ]

    rect = Rectangle(simplices, points)
    return Mesh(
            boundary_nodes,
            segment_n,
            interior_nodes=np.zeros((0, 2)),  # No need to pass interior nodes for rectangle mesh
            tri = None, # no triangulation
            nodes=points,
            boundary_ids=boundary_ids,
            corner_nodes=corner_nodes,
            adjacent_indices=adjacent_indices,
            type = "rect",
            rect=rect
        )

def triangulate(boundary_nodes, interior_nodes):
    # Make an array of all nodes
    nodes = np.vstack((boundary_nodes, interior_nodes))
    # Delaunay triangulation of everything
    tri = Delaunay(nodes)
    
    return tri, nodes

def get_corner_nodes(corners, boundary_nodes):
    # Find the coner nodes and their adjacent indices
    corner_nodes = np.array([i for i in range(len(boundary_nodes)) 
    if np.any(np.all(corners == boundary_nodes[i], axis=1))])

    # Find the adjacent edge indices
    adjacent_indices = np.zeros((len(corner_nodes), 2))
    # Loop over the corner indices
    for i in range(0, len(corner_nodes)) :
        # Find the index in corners of the node 
        corner_index = np.where(np.all(corners == boundary_nodes[corner_nodes[i]], axis=1))[0][0]
        # Since corners is counter clockwise, the two adjacent incides are just corner_index +- 1 (wrapping around 4)
        adjacent_indices[i] = np.array([(corner_index - 1) % 4, (corner_index)])
    return corner_nodes, adjacent_indices

def mesh_example(N_BOUNDARY = 4, N_INTERIOR = 1):
    # Make corners at desired places
    corners = np.array([[0,0],[2,0],[2,2],[0,2]])
    m = mesh(corners, N_BOUNDARY=N_BOUNDARY, N_INTERIOR=N_INTERIOR)

    return m

def mesh(corners, N_BOUNDARY = 4, N_INTERIOR = 1):
    # Generate all the nodes, then run the triangulation function
    (boundary_nodes, segment_n) = gen_boundary_nodes(corners, N_BOUNDARY)
    interior_nodes = gen_interior_nodes(corners, N_INTERIOR)
    (tri,nodes) = triangulate(boundary_nodes,interior_nodes)
    # Find all corner nodes and their adjacent indices
    (corner_nodes, adjacent_indices) = get_corner_nodes(corners, boundary_nodes)
    # Return separate and combine arrays, and triangle array
    return Mesh(
        boundary_nodes,
        segment_n,
        nodes,
        corner_nodes,
        adjacent_indices,
        type="tri",
        tri= tri
    )