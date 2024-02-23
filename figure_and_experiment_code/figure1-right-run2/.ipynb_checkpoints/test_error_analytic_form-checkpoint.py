import scipy.special as sc
from binary_search import binary_search,root_search


def get_risk_predictors(gamma,alpha):
    # I generator
    I_gen = lambda x,k, alpha: x*sc.hyp2f1(1,(1/alpha), 1 + (1/alpha), -k*x**alpha)
    # J generator
    J_gen = lambda x,k, alpha: x*sc.hyp2f1(2,(1/alpha), 1 + (1/alpha), -k*x**alpha)

    I = lambda k : I_gen(1/gamma, k, alpha) #\mathcal{I}
    J = lambda k : J_gen(1/gamma, k, alpha) #\mathcal{J}

    N = lambda k : 1 - I(k) # helper
    D = lambda k : 1 - J(k) # helper

    E_te = lambda k : 1/D(k) #\mathcal{E}_{\mathtt{test}}/\sigma^2
    E_tr = lambda k : N(k)**2/D(k) #\mathcal{E}_{\mathtt{train}}/\sigma^2
    R  = lambda k : k*(1-I(k)) # \mathcal{R}
    return E_te, E_tr, R


def get_E_tr_inv(gamma,alpha):
    _, E_tr, R = get_risk_predictors(gamma, alpha)
    k_crit = root_search(R)
    E_tr_inv = lambda tau : binary_search(E_tr, tau, init= k_crit)
    return E_tr_inv

# this is the grid of kappa's that we use in the paper
k_grid = [ 1.34,  1.99,  2.45,  2.92,  3.44,  4.03,  4.71,  5.5 ,  6.44,
        7.55,  8.9 , 10.54, 12.58, 15.15, 18.46, 22.8 , 28.67, 36.87,
       48.82, 67.2 ]

## Usage example

# E_te, E_tr, R = get_risk_predictors(gamma, alpha)
# gamma = 0.5
# alpha = 1.75
# r_grid = R(np.array(k_grid))