import numpy as np
from sklearn.neural_network import MLPRegressor

def run_MLP(X_tr,y_tr,X_te,y_te,E_tr):
    stopping_criterion = E_tr
    regr = MLPRegressor(random_state=1, batch_size=X_tr.shape[1])
    total_norms = []
    mse_losses_tr = []
    mse_losses_te = []
    for _ in range(500):
        regr = regr.partial_fit(X_tr.T, y_tr.squeeze())
        mse_loss_tr = np.mean((regr.predict(X_tr.T)-y_tr.squeeze())**2)
        mse_losses_tr.append(mse_loss_tr)
        total_norm = np.sum(np.array(list(map(np.linalg.norm,regr.coefs_)))**2)
        total_norms.append(total_norm)

        mse_loss_te = np.mean((regr.predict(X_te.T)-y_te.squeeze())**2)
        mse_losses_te.append(mse_loss_te)
        if stopping_criterion > mse_loss_tr:
            return {"mlp_norm":total_norm, "mlp_E_te": mse_loss_te, "mlp_converged": True}

    
    return {"mlp_norm":total_norms[-1], "mlp_E_te": mse_losses_te[-1], "mlp_converged": False}


def run_single_experiment(gamma, alpha, n_tr, n_te, E_tr, beta_scale = 10):
    p = int(n_tr/gamma) # data dimension
    
    idx = np.arange(1,p+1) # feature indices

    pop_evs = idx**(-alpha) # population level eigenvalues

    X_tr = np.multiply(np.sqrt(pop_evs[:,None]), np.random.normal(size= (p, n_tr)) )
    X_te = np.multiply(np.sqrt(pop_evs[:,None]), np.random.normal(size= (p, n_te)) )

    beta_true = np.sqrt(beta_scale)*np.random.normal(size= (p,1))/np.sqrt(p)

    y_tr = X_tr.T@beta_true + np.random.normal(size= (n_tr,1))
    y_te = X_te.T@beta_true + np.random.normal(size= (n_te,1))

    mlp_results = run_MLP(X_tr,y_tr,X_te,y_te,E_tr)
    return mlp_results
