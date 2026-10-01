from mesh import *
from gensystem import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# Desired parameters
W = 20
H = 30
N_INTERIOR = 400
N_BOUNDARY = 40
corners = np.array([[-W/2,-H/2],[W/2, -H/2],[W/2,H/2],[-W/2,H/2]])

# Define a load vector function
def f(x,y) :
    # return x*y
    return 100*(np.sin(x)+np.sin(y))

# Set boundary conditions going counter clockwise from the bottom edge
boundary_conditions = ['d', 'd', 'n', 'd']
q0 = 1
dirichlet_values = [10, 10, 10, 10]

mesh_data = mesh(corners, N_BOUNDARY=N_BOUNDARY, N_INTERIOR=N_INTERIOR)

tri = mesh_data.tri
nodes = mesh_data.nodes
boundary_nodes = mesh_data.boundary_nodes
interior_nodes = mesh_data.interior_nodes
segment_n = mesh_data.segment_n

dirichlet_n = np.array([
    i
    for i in range(len(boundary_nodes))
    if boundary_conditions[segment_n[i]] == 'd'
])

(K,b) = gen(tri, f, q0, boundary_nodes, segment_n, boundary_conditions, dirichlet_values)



# Solve for non dirichlet T
T_unknown = np.linalg.solve(K,b)

T = np.zeros(len(nodes))

n = 0
for i in range(0, len(nodes)):
    if i in dirichlet_n:
        T[i] = dirichlet_values[segment_n[i]]
    else:
        T[i] = T_unknown[n]
        n += 1
    

# plt.scatter(corners[:,0], corners[:,1])
# plt.scatter(nodes[:, 0], nodes[:, 1])
# for triangle in tri.simplices:
#     points = nodes[triangle]
#     points = np.vstack([points, points[0]])
#     plt.plot(points[:, 0], points[:, 1], 'k-')

# Convert SciPy Delaunay triangulation to Matplotlib triangulation
tri_plot = mtri.Triangulation(
    nodes[:, 0],
    nodes[:, 1],
    tri.simplices
)

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