# import matplotlib.pyplot as plt
def binary_search(f, tau, n_steps=20, init = 1.0, LB=0.0, UB=None):
    # Find an input rho to f
    # such that
    # f(rho) \approx tau
    # requires: f is monotonically increasing
    # Set UB to "None" for 
    rho = init
    
    for _ in range(n_steps):
        f_val = f(rho)
        # plt.axvline(x = rho)
        if f_val > tau:
            UB = rho
            rho = (rho+LB)/2
        else:
            LB = rho
            if UB is None:
                rho = 2*rho
            else:
                rho = (rho+UB)/2
    return rho

def root_search(f, n_steps=20, init = 1.0):
    # Find an input rho to f
    # such that
    # f(rho) \approx tau
    # requires: f is monotonically increasing
    rho = init
    UB = None
    LB = 0.0
    for _ in range(n_steps):
        f_val = f(rho)
        if f_val > 0:
            UB = rho
            rho = (rho+LB)/2
        else:
            LB = rho
            if UB is None:
                rho = 2*rho
            else:
                rho = (rho+UB)/2
    return rho