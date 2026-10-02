from mesh import *

def rect_mesh(corners, N_X, N_Y):

    (boundary_nodes, segment_n) = boundary_nodes(corners, N_X*2+N_Y*2-4)