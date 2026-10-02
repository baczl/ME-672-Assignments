from mesh import *
from gensystem import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import show3d

# Desired parameters :
# Taken from assignment 2
W = 20
H = 30
N_INTERIOR = 200
N_BOUNDARY = 40
N_X = 16
N_Y = 16

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
# To set all one:
# boundary_conditions = ['n'] *4
q0 = 1
dirichlet_values = [0, 10, 30, 15]


# OLD code for triangles

# mesh_data = mesh(corners, N_BOUNDARY=N_BOUNDARY, N_INTERIOR=N_INTERIOR)

# tri = mesh_data.tri
# nodes = mesh_data.nodes
# boundary_nodes = mesh_data.boundary_nodes
# segment_n = mesh_data.segment_n
# corner_nodes = mesh_data.corner_nodes
# adjacent_indices = mesh_data.adjacent_indices

# Create a rectangular mesh
rect_mesh = rectangle_mesh(corners, N_X, N_Y)

# Unpack rect_mesh data

nodes = rect_mesh.nodes
boundary_nodes = rect_mesh.boundary_nodes
segment_n = rect_mesh.segment_n
corner_nodes = rect_mesh.corner_nodes
adjacent_indices = rect_mesh.adjacent_indices
boundary_ids = rect_mesh.boundary_ids
rect = rect_mesh.rect
points = rect.points
simplices = rect.simplices

# Find dirichlet nodes
dirichlet_boundary_indices = np.array([
    i
    for i in range(len(boundary_nodes))
    if boundary_conditions[segment_n[i]] == 'd'
])

dirichlet_n = boundary_ids[dirichlet_boundary_indices]

# (K,b) = gen(tri, f, q0, boundary_nodes, segment_n, boundary_conditions, numpy.arange(0,N_BOUNDARY) dirichlet_values, corner_nodes, adjacent_indices, D)
(K, b) = gen_quad(rect_mesh, f, q0, boundary_conditions, dirichlet_values, D)

# Solve for non dirichlet T
T_unknown = np.linalg.solve(K,b)

T = np.zeros(len(nodes))


# Place the known T values back in the full T vector and put the unknown in their respective places
n = 0
for i in range(len(nodes)):

    if i in dirichlet_n:

        boundary_index = np.where(boundary_ids == i)[0][0]

        T[i] = dirichlet_values[
            int(segment_n[boundary_index])
        ]

    else:

        T[i] = T_unknown[n]
        n += 1
# # Convert SciPy Delaunay triangulation to Matplotlib triangulation
# tri_plot = mtri.Triangulation(
#     nodes[:, 0],
#     nodes[:, 1],
#     tri.simplices
# )

# show3d.display(tri_plot, T)

fig, ax = plt.subplots(figsize=(9, 7))

N = lambda xi, eta: np.array([
    (1-xi)*(1-eta)/4,
    (1+xi)*(1-eta)/4,
    (1+xi)*(1+eta)/4,
    (1-xi)*(1+eta)/4
])

# One global temperature scale
norm = Normalize(vmin=T.min(), vmax=T.max())
cmap = plt.get_cmap("coolwarm")

for element in simplices:

    # Global node coordinates for this quad
    p = points[element]

    # Global nodal temperatures for this quad
    Tq = T[element]

    # Sample the reference element
    xi = np.linspace(-1, 1, 30)
    eta = np.linspace(-1, 1, 30)

    X, Y = np.meshgrid(xi, eta)

    x = np.zeros_like(X)
    y = np.zeros_like(X)
    temp = np.zeros_like(X)

    for j in range(len(eta)):
        for i in range(len(xi)):

            n = N(X[j, i], Y[j, i])

            # Map reference coordinates -> physical coordinates
            x[j, i] = n @ p[:, 0]
            y[j, i] = n @ p[:, 1]

            # Q1 temperature interpolation
            temp[j, i] = n @ Tq

    # Plot this element using the GLOBAL temperature scale
    ax.contourf(
        x, y, temp,
        levels=np.linspace(T.min(), T.max(), 50),
        cmap=cmap,
        norm=norm
    )

# One colorbar for the entire solution
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(T)

fig.colorbar(sm, ax=ax, label="Temperature")

ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("FEM Temperature Distribution")

plt.tight_layout()
plt.show()