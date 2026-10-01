import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Develop a program that implements the Q1 Lagrange Finite Element
# method to solve the steady-state heat conduction in a solid. The pro
# gram should accept any rectangular domain with prescribed temper
# ature and/or heat flux boundary conditions. After testing, use the
# program to analyze the temperature distribution for the following case:
    # (a) Materials: Copper and MoS2, with anisotropy ratio kf/ks = 900,
    # where kf is the thermal conductivity along the fast axis and ks
    # along the slow axis.
    # (b) Domain: A 20mm × 30mm rectangular plate of thickness of
    # 1mm.
    # (c) Boundary conditions: Prescribed temperatures of 15◦C and
    # 10◦C on the long sides; and on the short sides, an inward heat flux
    # of 1W/mm2 and a prescribed temperature of 30◦C—cf. Figure 1.
    # The fast axis is aligned with Oy, and the slow axis with Ox.

"""Interactive boundary-condition input for a rectangular domain.

All BCs are stored internally in one general form:

    alpha * T + beta * (k dT/dn) = gamma

where n is the OUTWARD unit normal, so  k dT/dn = (k grad T) . n  is the
heat flux ENTERING the domain (W/mm^2).

    Dirichlet:  T = gamma                          (alpha = 1, beta = 0)
    Neumann:    beta * (k dT/dn) = gamma           (alpha = 0)
    Robin:      alpha * T + beta * (k dT/dn) = gamma
"""

SIDES = ("top", "right", "bottom", "left")
BC_TYPES = {"d": "dirichlet", "n": "neumann", "r": "robin"}

DIRECTIONS = {
    "dirichlet": (
        "  Dirichlet (prescribed temperature):  T = T_bar\n"
        "  Enter the temperature held along this side."
    ),
    "neumann": (
        "  Neumann (prescribed heat flux):  beta * (k dT/dn) = gamma\n"
        "  k dT/dn is the heat flux ENTERING the domain through this side.\n"
        "  For a plain inward flux q, enter beta = 1 and gamma = q\n"
        "  (use a negative q for heat leaving; q = 0 for an insulated side)."
    ),
    "robin": (
        "  Robin (mixed):  alpha * T + beta * (k dT/dn) = gamma\n"
        "  k dT/dn is the heat flux ENTERING the domain through this side.\n"
        "  For convection to a fluid at T_inf with coefficient h,\n"
        "  enter alpha = h, beta = 1, and gamma = h * T_inf."
    ),
}


def ask_float(prompt, nonzero=False):
    """Keep asking until the user enters a valid number."""
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("  Please enter a number.")
            continue
        if nonzero and value == 0.0:
            print("  This coefficient cannot be zero.")
            continue
        return value


def ask_bc_type(side):
    """Ask which BC type applies to one side."""
    while True:
        choice = input(
            f"\n{side.upper()} side: (d)irichlet, (n)eumann, or (r)obin? "
        ).strip().lower()
        if choice[:1] in BC_TYPES:  # accepts 'd', 'dir', 'dirichlet', ...
            return BC_TYPES[choice[:1]]
        print("  Please type d, n, or r.")


def get_boundary_conditions():
    """Return {side: {"type", "alpha", "beta", "gamma"}} for all four sides."""
    bcs = {}
    for side in SIDES:
        bc_type = ask_bc_type(side)
        print(DIRECTIONS[bc_type])

        if bc_type == "dirichlet":
            alpha, beta = 1.0, 0.0
            gamma = ask_float("  Temperature T_bar: ")
        elif bc_type == "neumann":
            alpha = 0.0
            beta = ask_float("  beta (coefficient on k dT/dn): ", nonzero=True)
            gamma = ask_float("  gamma (right-hand side): ")
        else:  # robin
            alpha = ask_float("  alpha (coefficient on T): ", nonzero=True)
            beta = ask_float("  beta (coefficient on k dT/dn): ", nonzero=True)
            gamma = ask_float("  gamma (right-hand side): ")

        bcs[side] = {"type": bc_type, "alpha": alpha, "beta": beta, "gamma": gamma}

    return bcs


if __name__ == "__main__":
    bcs = get_boundary_conditions()
    print("\nSummary:")
    for side, bc in bcs.items():
        print(f"  {side:>6}: {bc}")