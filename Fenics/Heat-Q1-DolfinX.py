"""
2D steady anisotropic heat conduction, Q1 Lagrange elements, in DOLFINx.
Same problem as run.py: -div(D grad T) = f on a W x H rectangle,
Neumann flux q0 on the bottom, Dirichlet on right/top/left.

Run inside the Docker container with: /dolfinx-env/bin/python3 heat_q1_dolfinx.py
(Written for DOLFINx 0.10/0.11 -- not tested here, check against your version.)
"""
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ---- Parameters (same as run.py) ----
W, H = 20.0, 30.0
nx, ny = 80, 60
ks = 1.0
kf = 900 * ks
boundary_conditions = ['n', 'd', 'd', 'd']   # bottom, right, top, left (CCW)
dirichlet_values = [0, 10, 30, 15]

# ---- Mesh: structured quads (this replaces mesh.py) ----
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-W / 2, -H / 2]), np.array([W / 2, H / 2])],
    [nx, ny],
    cell_type=mesh.CellType.quadrilateral,
)

# ---- Function space: Q1 Lagrange (this replaces galerkin.py) ----
V = fem.functionspace(domain, ("Lagrange", 1))

# ---- Tag the four edges so BCs and boundary integrals can find them ----
edge_markers = [
    lambda x: np.isclose(x[1], -H / 2),  # 0 bottom
    lambda x: np.isclose(x[0],  W / 2),  # 1 right
    lambda x: np.isclose(x[1],  H / 2),  # 2 top
    lambda x: np.isclose(x[0], -W / 2),  # 3 left
]
fdim = domain.topology.dim - 1
facets, tags = [], []
for k, marker in enumerate(edge_markers):
    found = mesh.locate_entities_boundary(domain, fdim, marker)
    facets.append(found)
    tags.append(np.full_like(found, k))
facets = np.hstack(facets)
tags = np.hstack(tags)
order = np.argsort(facets)
facet_tags = mesh.meshtags(domain, fdim, facets[order], tags[order])
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# ---- Weak form (this replaces weak_form.py) ----
D = ufl.as_matrix([[ks, 0.0], [0.0, kf]])
f = fem.Constant(domain, default_scalar_type(0.0))
q0 = fem.Constant(domain, default_scalar_type(1.0))

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(D * ufl.grad(T), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
for k, kind in enumerate(boundary_conditions):
    if kind == 'n':
        L += q0 * v * ds(k)   # positive q0 = heat flowing in (same as gensystem.py)

# ---- Dirichlet BCs (this replaces bound.py / dirichlet_bc) ----
bcs = []
for k, kind in enumerate(boundary_conditions):
    if kind == 'd':
        dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(k))
        bcs.append(fem.dirichletbc(default_scalar_type(dirichlet_values[k]), dofs, V))

# ---- Assemble and solve ----
problem = LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="heat_",
)
Th = problem.solve()

# ---- Plot (serial run) ----
xy = V.tabulate_dof_coordinates()[:, :2]
plt.figure(figsize=(9, 7))
cont = plt.tricontourf(xy[:, 0], xy[:, 1], Th.x.array, levels=40, cmap="coolwarm")
plt.colorbar(cont, label="Temperature")

geo = domain.geometry.x[:, :2]          # node coordinates
cells = domain.geometry.dofmap          # (n_cells, 4) node ids per quad
quads = geo[cells[:, [0, 1, 3, 2]]]     # reorder to go around the edge
plt.gca().add_collection(PolyCollection(
    quads, facecolors="none", edgecolors="black", linewidths=0.3, alpha=0.6))

plt.xlabel("x"); plt.ylabel("y"); plt.title("DOLFINx Q1 temperature")
plt.axis("equal"); plt.tight_layout()
plt.savefig("temperature_q1.png", dpi=150)
print("min/max T:", Th.x.array.min(), Th.x.array.max())