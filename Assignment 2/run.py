from mesh import *
from gensystem import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import show3d

# Desired parameters
W = 20
H = 30
N_INTERIOR = 1000
N_BOUNDARY = 4**4

corners = np.array([[-W/2,-H/2],[W/2, -H/2],[W/2,H/2],[-W/2,H/2]])

# Heat conduction parameters
ks = 1
kf = 900*ks

# Define heat conduction tensor
D = np.array([
    [ks, 0],
    [0, kf]
])

# Define a load vector function
def f(x,y) :
    return 0

# Set boundary conditions going counter clockwise from the bottom edge
boundary_conditions = ['n', 'd', 'd', 'd']
q0 = 1
dirichlet_values = [0, 10, 30, 15]

# mesh_data = mesh(corners, N_BOUNDARY=N_BOUNDARY, N_INTERIOR=N_INTERIOR)

mesh_data = mesh(corners, N_BOUNDARY=N_BOUNDARY, N_INTERIOR=N_INTERIOR)

tri = mesh_data.tri
nodes = mesh_data.nodes
boundary_nodes = mesh_data.boundary_nodes
interior_nodes = mesh_data.interior_nodes
segment_n = mesh_data.segment_n
corner_nodes = mesh_data.corner_nodes
adjacent_indices = mesh_data.adjacent_indices


dirichlet_n = np.array([
    i
    for i in range(len(boundary_nodes))
    if boundary_conditions[segment_n[i]] == 'd'
])

(K,b) = gen(tri, f, q0, boundary_nodes, segment_n, boundary_conditions, dirichlet_values, corner_nodes, adjacent_indices, D)



# Solve for non dirichlet T
T_unknown = np.linalg.solve(K,b)

T = np.zeros(len(nodes))


# Place the known T values back in the full T vector and put the unknown in their respective places
n = 0
for i in range(0, len(nodes)):
    # If the node is a dirichlet node, use a known value.
    if i in dirichlet_n:
        # If the node is on a corner, average the two dirichlet values.
        if np.any(corner_nodes == i):
            corner_index = np.where(corner_nodes == i)[0][0]
            T[i] = 1/2*(dirichlet_values[int(adjacent_indices[corner_index][0])] + dirichlet_values[int(adjacent_indices[corner_index][1])])
        else:
            # Otherwise just use the dirichlet value for that edge.
            T[i] = dirichlet_values[segment_n[i]]
    else:
        # If the node is not a dirichlet node, use the value from the solution.
        T[i] = T_unknown[n]
        # Keep a count for unknown to put them in the right spots. So i increases every
        # loop but n only increases if we use up an unknown value.
        n += 1
    
# Convert SciPy Delaunay triangulation to Matplotlib triangulation
tri_plot = mtri.Triangulation(
    nodes[:, 0],
    nodes[:, 1],
    tri.simplices
)

show3d.display(tri_plot, T)

# Create figure
plt.figure(figsize=(9, 7))

# Plot temperature field
contour = plt.tripcolor(
    tri_plot,
    T,
    shading='gouraud',
    cmap='coolwarm'
)

# Plot mesh
plt.triplot(
    tri_plot,
    color='black',
    linewidth=0.3,
    alpha=0.5
)

# Plot nodes
plt.scatter(
    nodes[:, 0],
    nodes[:, 1],
    color='black',
    s=8
)

# Colorbar
plt.colorbar(
    contour,
    label='Temperature'
)

# Labels
plt.xlabel('x')
plt.ylabel('y')
plt.title('FEM Temperature Distribution')

# Preserve physical aspect ratio
plt.axis('equal')

plt.tight_layout()
plt.show()