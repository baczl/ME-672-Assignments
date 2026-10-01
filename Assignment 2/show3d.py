import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def display(tri_plot, T):
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot temperature as z
    surface = ax.plot_trisurf(
        tri_plot,
        T,
        cmap='coolwarm',
        edgecolor='black',
        linewidth=0.2
    )

    # Colorbar
    fig.colorbar(
        surface,
        ax=ax,
        label='Temperature'
    )

    # Axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    ax.set_title('FEM Temperature Distribution')

    # Adjust viewing angle
    ax.view_init(elev=30, azim=-135)

    plt.tight_layout()
    plt.show()