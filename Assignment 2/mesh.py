# Needed for matrix algebra
import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay

from dataclasses import dataclass

@dataclass
class Mesh:
    boundary_nodes: np.ndarray
    segment_n: np.ndarray
    interior_nodes: np.ndarray
    tri: Delaunay
    nodes: np.ndarray

def gen_boundary_nodes(corners, N_BOUNDARY=4):
    segment_n = [0] * N_BOUNDARY
    # If there are four nodes, just put them at the corners
    if N_BOUNDARY == 4:
        boundary_nodes = corners
        segment_n = [0, 1, 2, 3]
    else:
        # Calculate total perimeter - equal to 4 times the x and y of the top corner
        perimeter = corners[2][0]*4+corners[2][1]*4
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

def gen_interior_nodes(corners, N_INTERIOR = 1):
    if N_INTERIOR == 1:
        interior_nodes = np.array([0,0])
    else:
        # Otherwise just generate randoms inside boundary
        rng = np.random.default_rng()
    
        interior_nodes = np.column_stack((  

        rng.uniform(low=corners[0,0], high=corners[2,0], size=N_INTERIOR),
        rng.uniform(low=corners[0,1], high=corners[2,1], size=N_INTERIOR)))

    return interior_nodes

def triangulate(boundary_nodes, interior_nodes):
    # Make an array of all nodes
    nodes = np.vstack((boundary_nodes, interior_nodes))
    # Delaunay triangulation of everything
    tri = Delaunay(nodes)
    
    return tri, nodes

def mesh(corners, N_BOUNDARY = 4, N_INTERIOR = 1):
    # Generate all the nodes, then run the triangulation function
    (boundary_nodes, segment_n) = gen_boundary_nodes(corners, N_BOUNDARY)
    interior_nodes = gen_interior_nodes(corners, N_INTERIOR)
    (tri,nodes) = triangulate(boundary_nodes,interior_nodes)

    # Return separate and combine arrays, and triangle array
    return Mesh(
        boundary_nodes,
        segment_n,
        interior_nodes,
        tri,
        nodes,
    )