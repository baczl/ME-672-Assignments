"""Symbolic strong-to-weak form derivation for 2D anisotropic heat conduction.

Strong form on the rectangle  Omega = (0, Lx) x (0, Ly):

    -d/dx( kx dT/dx ) - d/dy( ky dT/dy ) = Q      in Omega

Boundary conditions come from the `bcs` dictionary produced by
boundary_conditions.get_boundary_conditions(), in the general form

    alpha * T + beta * (k dT/dn) = gamma          on each side

with n the OUTWARD normal (k dT/dn = heat flux ENTERING the domain).
"""

import sympy as sp

# ---------------------------------------------------------------------------
# Symbols and unknown functions
# ---------------------------------------------------------------------------
x, y = sp.symbols("x y", real=True)
Lx, Ly, kx, ky = sp.symbols("L_x L_y k_x k_y", positive=True)
Q = sp.Symbol("Q", real=True)              # internal heat source (0 here)

T = sp.Function("T")(x, y)                 # trial function (temperature)
v = sp.Function("v")(x, y)                 # test function

qx = kx * sp.diff(T, x)                    # conductive flux terms
qy = ky * sp.diff(T, y)

# Geometry of each side: which coordinate is fixed, where, the coordinate
# you integrate along, its limits, and the sign of the outward normal.
SIDE_INFO = {
    "right":  {"fixed": x, "at": Lx, "along": y, "lims": (0, Ly), "n": +1, "flux": qx},
    "left":   {"fixed": x, "at": 0,  "along": y, "lims": (0, Ly), "n": -1, "flux": qx},
    "top":    {"fixed": y, "at": Ly, "along": x, "lims": (0, Lx), "n": +1, "flux": qy},
    "bottom": {"fixed": y, "at": 0,  "along": x, "lims": (0, Lx), "n": -1, "flux": qy},
}


def _num(value):
    """Turn a float like 1.0 or 0.5 into a clean SymPy number (1, 1/2)."""
    return sp.nsimplify(value, rational=True)


def _on_side(expr, side):
    """Restrict an expression to one side, e.g. T(x, y) -> T(x, 0).

    Derivatives are wrapped in an unevaluated Subs so they display as
    (dT/dx) evaluated at x = Lx, instead of a derivative w.r.t. Lx.
    """
    info = SIDE_INFO[side]
    fixed, at = info["fixed"], info["at"]
    expr = expr.replace(lambda e: isinstance(e, sp.Derivative),
                        lambda e: sp.Subs(e, fixed, at))
    return expr.subs(fixed, at)


def _side_integral(integrand, side):
    """Line integral of `integrand` along one side of the rectangle."""
    info = SIDE_INFO[side]
    a, b = info["lims"]
    return sp.Integral(_on_side(integrand, side), (info["along"], a, b))


def derive_weak_form(bcs, source=0, verbose=True):
    """Derive the weak form step by step and return the pieces.

    Parameters
    ----------
    bcs : dict
        {side: {"type", "alpha", "beta", "gamma"}} for top/right/bottom/left.
    source : number
        Value of the heat source Q (0 for the assignment).
    verbose : bool
        Print each step of the derivation.
    """
    say = (lambda e: sp.pprint(e, wrap_line=False)) if verbose else (lambda e: None)
    log = print if verbose else (lambda *a, **k: None)
    area = ((x, 0, Lx), (y, 0, Ly))

    # -- Step 1: strong form ------------------------------------------------
    strong_lhs = -sp.diff(qx, x) - sp.diff(qy, y)
    strong = sp.Eq(strong_lhs, Q)
    log("\nSTEP 1  Strong form (in Omega):")
    say(strong)

    # -- Step 2: multiply by v and integrate --------------------------------
    step2 = sp.Eq(sp.Integral(v * strong_lhs, *area), sp.Integral(Q * v, *area))
    log("\nSTEP 2  Multiply by test function v and integrate over Omega:")
    say(step2)

    # -- Step 3: integration by parts (product rule), verified --------------
    # v * dF/dx = d(vF)/dx - (dv/dx) F   for F = qx, and likewise in y
    for F, var in ((qx, x), (qy, y)):
        lhs_id = v * sp.diff(F, var)
        rhs_id = sp.diff(v * F, var) - sp.diff(v, var) * F
        assert sp.simplify(lhs_id - rhs_id) == 0, "product rule check failed"
    log("\nSTEP 3  Integrate by parts in x and y (product rule verified):")
    log("        v dF/dx = d(vF)/dx - (dv/dx) F")

    volume_integrand = kx * sp.diff(v, x) * sp.diff(T, x) + ky * sp.diff(v, y) * sp.diff(T, y)
    volume = sp.Integral(volume_integrand, *area)

    # Each boundary term is  - integral over side of  v * (k dT/dn)
    raw_boundary = 0
    for side, info in SIDE_INFO.items():
        raw_boundary += _side_integral(-info["n"] * v * info["flux"], side)

    step3 = sp.Eq(volume + raw_boundary, sp.Integral(Q * v, *area))
    say(step3)

    # -- Step 4: apply the boundary conditions ------------------------------
    log("\nSTEP 4  Apply boundary conditions:")
    lhs = volume
    rhs = sp.Integral(source * v, *area) if source != 0 else sp.Integer(0)
    dirichlet_sides, essential = [], []

    for side in ("bottom", "right", "top", "left"):
        bc = bcs[side]
        alpha, beta, gamma = _num(bc["alpha"]), _num(bc["beta"]), _num(bc["gamma"])

        if bc["type"] == "dirichlet":
            dirichlet_sides.append(side)
            essential.append(sp.Eq(_on_side(T, side), gamma))
            log(f"  {side:>6}: Dirichlet, T = {gamma}  ->  v = 0 here, term vanishes")

        elif bc["type"] == "neumann":
            # k dT/dn = gamma/beta  ->  -int v*(gamma/beta)  moves to RHS
            if gamma != 0:  # insulated side (gamma = 0) adds nothing
                rhs += _side_integral((gamma / beta) * v, side)
            log(f"  {side:>6}: Neumann, k dT/dn = {gamma / beta}  ->  load term on RHS")

        else:  # robin
            # k dT/dn = (gamma - alpha*T)/beta
            lhs += _side_integral((alpha / beta) * T * v, side)
            if gamma != 0:
                rhs += _side_integral((gamma / beta) * v, side)
            log(f"  {side:>6}: Robin, k dT/dn = ({gamma} - {alpha} T)/{beta}"
                "  ->  terms on both sides")

    weak = sp.Eq(lhs, rhs)

    # -- Step 5: final statement --------------------------------------------
    log("\nSTEP 5  Weak form:")
    if dirichlet_sides:
        log(f"  Find T in H^1(Omega) with T fixed on {', '.join(dirichlet_sides)}:")
        for cond in essential:
            say(cond)
        log(f"  such that, for all v in H^1(Omega) with v = 0 on {', '.join(dirichlet_sides)},")
    else:
        log("  Find T in H^1(Omega) such that, for all v in H^1(Omega),")
    say(weak)

    return {
        "strong": strong,
        "weak": weak,
        "lhs": lhs,
        "rhs": rhs,
        "T": T,
        "v": v,
        "bcs": bcs,
        "essential_bcs": essential,
        "dirichlet_sides": dirichlet_sides,
        "latex_strong": sp.latex(strong),
        "latex_weak": sp.latex(weak),
    }


if __name__ == "__main__":
    # Ask for the BCs interactively, then derive the weak form from them
    from bound import get_boundary_conditions

    bcs = get_boundary_conditions()
    result = derive_weak_form(bcs)
    print("\nLaTeX of weak form:\n", result["latex_weak"])