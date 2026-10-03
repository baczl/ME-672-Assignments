from mesh import *
from gensystem import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# import show3d


# ---------------- Input helpers ----------------
# Functions to help with the user input sections. 
def ask_float(prompt, min_val=None):
    while True:
        try:
            val = float(input(prompt))
        except ValueError:
            print("  Please enter a number.")
            continue
        if min_val is not None and val <= min_val:
            print(f"  Value must be greater than {min_val}.")
            continue
        return val

def ask_int(prompt, lo, hi):
    while True:
        try:
            val = int(input(prompt))
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if not lo <= val <= hi:
            print(f"  Must be between {lo} and {hi}.")
            continue
        return val

def ask_choice(prompt, options):
    while True:
        val = input(prompt).strip().lower()
        if val in options:
            return val
        print(f"  Choose one of: {', '.join(options)}")

def ask_conductivity():
    print("\n--- Thermal conductivity ---")
    k_type = ask_choice("Single coefficient or tensor? (s/t): ", ["s", "t"])

    if k_type == "s":
        k = ask_float("  Thermal conductivity k: ", min_val=0)
        return np.array([[k, 0],
                         [0, k]])

    while True:
        kxx = ask_float("  kxx: ")
        kyy = ask_float("  kyy: ")
        kxy = ask_float("  kxy: ")
        kyx = ask_float("  kyx: ")
        D = np.array([[kxx, kxy],
                      [kyx, kyy]])

        # Positive definiteness is checked on the symmetric part
        if np.any(np.linalg.eigvalsh(0.5 * (D + D.T)) <= 0):
            print("  Tensor is not positive definite (need kxx > 0 and kxx*kyy > kxy^2). Try again.\n")
            continue

        if not np.isclose(kxy, kyx):
            print("  Warning: kxy != kyx. Physical conductivity tensors are symmetric;"
                  " the non-symmetric part only adds boundary terms here.")
        return D

def print_tensor(D):
    print("\nConductivity tensor D =")
    print(f"  | {D[0,0]:>10.4g}  {D[0,1]:>10.4g} |")
    print(f"  | {D[1,0]:>10.4g}  {D[1,1]:>10.4g} |\n")

# ---------------- Case selection ----------------
MAX_ELEMENTS = 30   # per direction; dblquad assembly gets slow past this
EDGE_NAMES = ["bottom", "right", "top", "left"]

case = ask_choice("Choose case ('test' or 'general'): ", ["test", "general"])

if case == "test":
    # Original Assignment 2 values
    W, H = 20, 30
    # Cartesian mesh: one element per unit length
    n_el_x = round(W)   # 20 elements
    n_el_y = round(H)   # 30 elements
    N_X = n_el_x + 1
    N_Y = n_el_y + 1
    boundary_conditions = ['n', 'd', 'd', 'd']
    dirichlet_values = [0, 10, 30, 15]
    q_values = [1, 0, 0, 0]
    D = np.array([[1, 0],
                  [0, 900]])   # ks = 1, kf = 900*ks    

else:
    # ---- Domain and mesh ----
    print("\n--- Domain and mesh ---")
    W = ask_float("Domain width W: ", min_val=0)
    H = ask_float("Domain height H: ", min_val=0)

    mesh_type = ask_choice("Mesh type: cartesian (1 element per unit length) or user-defined? (c/u): ",
                           ["c", "u"])

    if mesh_type == "c":
        n_el_x = max(1, round(W))
        n_el_y = max(1, round(H))

        if not (np.isclose(W, n_el_x) and np.isclose(H, n_el_y)):
            print(f"  Note: W and H aren't whole numbers; element size will be "
                  f"{W/n_el_x:.4g} x {H/n_el_y:.4g}.")

        if n_el_x > MAX_ELEMENTS or n_el_y > MAX_ELEMENTS:
            print(f"  A cartesian mesh needs {n_el_x} x {n_el_y} elements, which exceeds the "
                  f"{MAX_ELEMENTS}-per-direction limit. Switching to user-defined.")
            mesh_type = "u"
        else:
            print(f"  Using a {n_el_x} x {n_el_y} cartesian mesh.")

    if mesh_type == "u":
        n_el_x = ask_int(f"Number of elements in x (1-{MAX_ELEMENTS}): ", 1, MAX_ELEMENTS)
        n_el_y = ask_int(f"Number of elements in y (1-{MAX_ELEMENTS}): ", 1, MAX_ELEMENTS)

    N_X = n_el_x + 1   # nodes = elements + 1
    N_Y = n_el_y + 1
    D = ask_conductivity()

    # ---- Boundary conditions ----
    print("\n--- Boundary conditions (start at bottom, go counterclockwise) ---")
    while True:
        boundary_conditions, dirichlet_values, q_values = [], [], []
        for name in EDGE_NAMES:
            bc = ask_choice(f"{name} edge: Dirichlet or Neumann? (d/n): ", ["d", "n"])
            boundary_conditions.append(bc)
            if bc == 'd':
                dirichlet_values.append(ask_float(f"  {name} temperature T [°C]: "))
                q_values.append(0.0)
            else:
                q_values.append(ask_float(f"  {name} heat flux q: "))
                dirichlet_values.append(0.0)
        if 'd' in boundary_conditions:
            break
        print("  At least one edge must be Dirichlet (all-Neumann has no unique solution). Try again.\n")

corners = np.array([[-W/2, -H/2], [W/2, -H/2], [W/2, H/2], [-W/2, H/2]])

print_tensor(D)
show_elements = ask_choice("Show element edges on the plot? (y/n): ", ["y", "n"]) == "y"

# Define a load vector function
def f(x,y) :
    return 0

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
(K, b) = gen_quad(rect_mesh, f, q_values, boundary_conditions, dirichlet_values, D)

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
        # Optionally outline this element
    if show_elements:
        outline = np.vstack([p, p[0]])   # close the loop back to the first node
        ax.plot(outline[:, 0], outline[:, 1], color="k", linewidth=0.5)

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