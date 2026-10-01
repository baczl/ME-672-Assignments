from bound import get_boundary_conditions
from weak_form import derive_weak_form
from galerkin import galerkin_discretize

bcs = get_boundary_conditions()
weak = derive_weak_form(bcs)
gal = galerkin_discretize(weak)

def main():
    bcs = get_boundary_conditions()
    result = derive_weak_form(bcs)

if __name__ == "__main__":
    main()