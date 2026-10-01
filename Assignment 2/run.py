from mesh import *
import matplotlib.pyplot as plt

# Desired parameters
W = 10
H = 40
N_INTERIOR = 18
N_BOUNDARY = 16
corners = np.array([[-W/2,-H/2],[W/2, -H/2],[W/2,H/2],[-W/2,H/2]])

(boundary_nodes, interior_nodes, tri, nodes) = mesh(corners, N_BOUNDARY=N_BOUNDARY, N_INTERIOR=N_INTERIOR)

plt.scatter(corners[:,0], corners[:,1])
plt.scatter(nodes[:, 0], nodes[:, 1])
for triangle in tri.simplices:
    points = nodes[triangle]
    points = np.vstack([points, points[0]])
    plt.plot(points[:, 0], points[:, 1], 'k-')

plt.axis('equal')
plt.show()