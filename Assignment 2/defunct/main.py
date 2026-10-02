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

from bound import get_boundary_conditions
from weak_form import derive_weak_form
from galerkin import galerkin_discretize

def main():
    bcs = get_boundary_conditions()
    weak = derive_weak_form(bcs)
    gal = galerkin_discretize(weak)

if __name__ == "__main__":
    main()