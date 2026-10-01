"""Weak form -> Galerkin system -> Q1 element formulas.

Takes the dictionary returned by weak_form.derive_weak_form() and:

  1. Substitutes  T_h = sum_j q_j phi_j  and  v = phi_i  into the weak form,
     giving the discrete system  sum_j K_ij q_j = f_i.
  2. Specializes K_ij and f_i to ONE bilinear (Q1) rectangular element,
     computing the 4x4 element stiffness matrix exactly with SymPy.
  3. Derives the edge vectors/matrices needed for Neumann and Robin sides.
  4. Converts everything to fast NumPy functions for the assembly step.

Q1 local node order (counterclockwise, matches the mesh connectivity):

      3 ---- 2        reference square  -1 <= xi, eta <= 1
      |      |
      0 ---- 1
"""

import numpy as np
import sympy as sp

from weak_form import x, y, kx, ky

# Element size and reference coordinates
hx, hy, h = sp.symbols("h_x h_y h", positive=True)
xi, eta, s = sp.symbols("xi eta s", real=True)
Qs = sp.Symbol("Q", real=True)                  # heat source per volume
g = sp.Symbol("g", real=True)                   # edge value (gamma/beta etc.)

# Reference coordinates of the four local nodes
Q1_NODES = ((-1, -1), (1, -1), (1, 1), (-1, 1))


def _show(label, expr, verbose):
    if verbose:
        print(label)
        sp.pprint(expr, wrap_line=False)


# ---------------------------------------------------------------------------
# Step 1: the general Galerkin system
# ---------------------------------------------------------------------------
def galerkin_system(weak_result, verbose=True):
    """Substitute T_h = sum q_j phi_j, v = phi_i and extract K_ij and f_i."""
    T, v = weak_result["T"], weak_result["v"]
    lhs, rhs = weak_result["lhs"], weak_result["rhs"]

    phi_i = sp.Function("phi_i")(x, y)
    phi_j = sp.Function("phi_j")(x, y)
    q = sp.IndexedBase("q")
    j, n = sp.symbols("j n", integer=True, positive=True)

    if verbose:
        print("\nGALERKIN STEP 1  Approximate T and choose v from the same basis:")
    _show("", sp.Eq(sp.Symbol("T_h"), sp.Sum(q[j] * phi_j, (j, 1, n))), verbose)
    if verbose:
        print("  and test with v = phi_i for i = 1, ..., n.")

    # The weak form is linear in T, so the sum and q_j come out of every
    # integral; what is left multiplying q_j is K_ij.
    K_ij = lhs.replace(T.func, phi_j.func).replace(v.func, phi_i.func)
    f_i = rhs.replace(v.func, phi_i.func) if rhs != 0 else sp.Integer(0)

    if verbose:
        print("\nGALERKIN STEP 2  Discrete system  sum_j K_ij q_j = f_i,  with")
    _show("K_ij =", K_ij, verbose)
    _show("f_i =", f_i, verbose)

    return {"K_ij": K_ij, "f_i": f_i,
            "latex_K": sp.latex(K_ij), "latex_f": sp.latex(f_i)}


# ---------------------------------------------------------------------------
# Step 2: Q1 element formulas
# ---------------------------------------------------------------------------
def q1_shape_functions():
    """Bilinear shape functions N_a(xi, eta) on the reference square."""
    return [sp.Rational(1, 4) * (1 + xa * xi) * (1 + ea * eta) for xa, ea in Q1_NODES]


def q1_element_stiffness(verbose=True):
    """Exact 4x4 element stiffness for a rectangle of size hx by hy.

    K^e_ab = integral over element of  kx dNa/dx dNb/dx + ky dNa/dy dNb/dy
    with  x = x0 + hx(1+xi)/2,  y = y0 + hy(1+eta)/2,  dx dy = (hx hy / 4) dxi deta.
    """
    N = q1_shape_functions()
    detJ = hx * hy / 4
    dN_dx = [sp.diff(Na, xi) * 2 / hx for Na in N]      # chain rule
    dN_dy = [sp.diff(Na, eta) * 2 / hy for Na in N]

    Ke = sp.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            integrand = (kx * dN_dx[a] * dN_dx[b] + ky * dN_dy[a] * dN_dy[b]) * detJ
            Ke[a, b] = sp.simplify(sp.integrate(integrand, (xi, -1, 1), (eta, -1, 1)))

    if verbose:
        print("\nQ1 STEP 1  Bilinear shape functions on the reference square:")
        for a, Na in enumerate(N):
            sp.pprint(sp.Eq(sp.Symbol(f"N_{a}"), sp.factor(Na)), wrap_line=False)
        print("\nQ1 STEP 2  Element stiffness matrix K^e (exact integration):")
        sp.pprint(Ke, wrap_line=False)
        # Sanity check: each row must sum to zero (a constant T gives no flux)
        print("  Row sums (should all be 0):", [sp.simplify(sum(Ke.row(r))) for r in range(4)])
    return Ke


def q1_element_source():
    """Element load vector from a uniform source Q: integral of Q N_a."""
    N = q1_shape_functions()
    detJ = hx * hy / 4
    return sp.Matrix([sp.integrate(Qs * Na * detJ, (xi, -1, 1), (eta, -1, 1)) for Na in N])


def q1_edge_terms(verbose=True):
    """Edge vector (Neumann/Robin load) and edge matrix (Robin) for one edge.

    On an edge of length h the two nonzero shape functions are linear:
      N_1 = (1 - s)/2,  N_2 = (1 + s)/2,  ds = (h/2) ds_ref
    """
    Ne = [(1 - s) / 2, (1 + s) / 2]
    jac = h / 2
    f_edge = sp.Matrix([sp.integrate(g * Na * jac, (s, -1, 1)) for Na in Ne])
    M_edge = sp.Matrix(2, 2, lambda a, b: sp.integrate(Ne[a] * Ne[b] * jac, (s, -1, 1)))

    if verbose:
        print("\nQ1 STEP 3  Edge terms (g = gamma/beta, c = alpha/beta):")
        print("  Neumann/Robin load vector on an edge:  f_edge = integral of g N_a ds")
        sp.pprint(f_edge, wrap_line=False)
        print("  Robin matrix on an edge:  K_edge = c * integral of N_a N_b ds,  with")
        sp.pprint(M_edge, wrap_line=False)
    return f_edge, M_edge


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def galerkin_discretize(weak_result, verbose=True):
    """Run the full weak -> Galerkin -> Q1 step and return symbolic + numeric forms.

    Numeric functions in the returned dict (all return NumPy arrays):
      element_stiffness(kx, ky, hx, hy)  -> (4, 4)
      element_source(Q, hx, hy)          -> (4,)
      edge_load(g, h)                    -> (2,)   g = gamma/beta
      edge_mass(h)                       -> (2, 2) multiply by alpha/beta
    """
    system = galerkin_system(weak_result, verbose)
    Ke = q1_element_stiffness(verbose)
    fe_src = q1_element_source()
    f_edge, M_edge = q1_edge_terms(verbose)

    return {
        **system,
        "Ke": Ke,
        "fe_source": fe_src,
        "f_edge": f_edge,
        "M_edge": M_edge,
        "latex_Ke": sp.latex(Ke),
        "element_stiffness": sp.lambdify((kx, ky, hx, hy), Ke, "numpy"),
        "element_source": lambda Q, hx_, hy_: np.full(4, Q * hx_ * hy_ / 4),
        "edge_load": lambda g_, h_: np.full(2, g_ * h_ / 2),
        "edge_mass": lambda h_: (h_ / 6) * np.array([[2.0, 1.0], [1.0, 2.0]]),
    }


if __name__ == "__main__":
    from bound import get_boundary_conditions
    from weak_form import derive_weak_form

    bcs = get_boundary_conditions()
    weak = derive_weak_form(bcs)
    gal = galerkin_discretize(weak)

    # Quick numeric check: isotropic unit square element
    print("\nNumeric K^e for kx = ky = 1, hx = hy = 1:")
    print(np.array(gal["element_stiffness"](1.0, 1.0, 1.0, 1.0)))