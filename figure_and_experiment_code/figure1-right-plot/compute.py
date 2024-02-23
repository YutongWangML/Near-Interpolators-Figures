#!/usr/bin/env python
# coding: utf-8

# In[218]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from test_error_analytic_form import get_risk_predictors, k_grid, get_E_tr_inv

import vlab
config = vlab.utils.load_yaml_file('config.yaml')


# In[219]:


hls = eval(config['hidden_layer_sizes'])
hls
n_tr = config['n_tr']
n_trs = eval(config['n_trs'])
alpha = config['alpha']
gamma = eval(config['gamma'])
taus = eval(config['taus'])
tau = config['tau']
n_te = config['n_te']
max_iter = config['max_iter']
replicate = config['replicate']
method = config['method']
sweep = config['sweep']
E_te, E_tr, R = get_risk_predictors(gamma=gamma, alpha=alpha)
E_tr_inv = get_E_tr_inv(gamma=gamma, alpha=alpha)
print("method:", method)
print("sweep:", sweep)
print("n_tr:", n_tr)
print("tau:", tau)


# In[220]:


n_trs =  np.logspace(2.302,2.7,10).astype(int)
n_trs


# In[221]:


n_tr = 1000


# In[222]:


import sys

def is_ipykernel():
    return any('ipykernel_launcher.py' in arg for arg in sys.argv)

if is_ipykernel():
    print("Running in ipykernel mode!")
    job_id = 1
else:
    print("Not running in ipykernel mode.")
    job_id = sys.argv[1]
print(job_id)


# In[223]:


def run_MLP(X_tr,y_tr,X_te,y_te,E_tr):
    # run multilayer perceptron
    stopping_criterion = E_tr
    regr = MLPRegressor(hidden_layer_sizes=hls,random_state=1, batch_size=X_tr.shape[1])
    total_norms = []
    mse_losses_tr = []
    mse_losses_te = []
    for _ in range(max_iter):
        regr = regr.partial_fit(X_tr.T, y_tr.squeeze())
        mse_loss_tr = np.mean((regr.predict(X_tr.T)-y_tr.squeeze())**2)
        mse_losses_tr.append(mse_loss_tr)
        total_norm = np.sum(np.array(list(map(np.linalg.norm,regr.coefs_)))**2)
        total_norms.append(total_norm)

        mse_loss_te = np.mean((regr.predict(X_te.T)-y_te.squeeze())**2)
        mse_losses_te.append(mse_loss_te)
        if stopping_criterion > mse_loss_tr:
            return {"mlp_norm":total_norm, "mlp_E_te": mse_loss_te, "mlp_E_tr": mse_loss_tr, "mlp_converged": True}

    
    return {"norm":total_norms[-1], "E_te": mse_losses_te[-1], "mlp_converged": False}

def run_RREG(X_tr,y_tr,X_te,y_te,E_tr):
    # n_tr = len(y_tr)
    p,n_tr = X_tr.shape
    hatSig = (1/n_tr)*X_tr@X_tr.T # sample covariance matrix

    r = R(E_tr_inv(E_tr))
    # beta from ridge regression
    beta_rreg = (1/n_tr)*np.linalg.solve(hatSig + r*n_tr**(-alpha)*np.eye(p), X_tr@y_tr)

    # calculate ridge regression stats
    norm = np.linalg.norm(beta_rreg)**2
    E_tr = np.mean(np.square(y_tr-X_tr.T@beta_rreg))
    E_te = np.mean(np.square(y_te-X_te.T@beta_rreg))
    return {"norm": norm, "E_te": E_te, "E_tr": E_tr}

def run_single_experiment(gamma, alpha, n_tr, n_te, E_tr, method, beta_scale = 10):
    p = int(n_tr/gamma) # data dimension
    
    idx = np.arange(1,p+1) # feature indices

    pop_evs = idx**(-alpha) # population level eigenvalues

    X_tr = np.multiply(np.sqrt(pop_evs[:,None]), np.random.normal(size= (p, n_tr)) )
    X_te = np.multiply(np.sqrt(pop_evs[:,None]), np.random.normal(size= (p, n_te)) )

    beta_true = np.sqrt(beta_scale)*np.random.normal(size= (p,1))/np.sqrt(p)

    y_tr = X_tr.T@beta_true + np.random.normal(size= (n_tr,1))
    y_te = X_te.T@beta_true + np.random.normal(size= (n_te,1))

    if method == "rreg":
        result = run_RREG(X_tr,y_tr,X_te,y_te,E_tr)
    elif method == "MLP":
        result = run_MLP(X_tr,y_tr,X_te,y_te,E_tr)
    result['n_tr'] = n_tr
    result['tau'] = E_tr
    return result


# In[224]:


if sweep == "tau":
    print([(n_tr,tau) for tau in taus for _ in range(replicate)])
elif sweep == "n_tr":
    print([(n_tr,tau) for n_tr in n_trs for _ in range(replicate)])
else:
    raise Exception("Undefined sweep type")


# In[225]:


if sweep == "tau":
    results = [run_single_experiment(alpha=alpha,gamma=gamma, n_tr=n_tr,n_te = n_te, E_tr= tau,method=method) for tau in taus for _ in range(replicate)]
elif sweep == "n_tr":
    results = [run_single_experiment(alpha=alpha,gamma=gamma, n_tr=n_tr,n_te = n_te, E_tr= tau,method=method) for n_tr in n_trs for _ in range(replicate)]
else:
    raise Exception("Undefined sweep type")


# In[226]:


df = pd.DataFrame(results)
df.to_csv(f'outputs/results{job_id}.csv')


# In[227]:


df


# In[228]:


plt.scatter(df['E_tr'], df['E_te'])
from binary_search import binary_search
gamma = 0.75
alpha = 1.75
colors = ['#1f77b4', '#419ede']
def plot_trade_off_curve(gamma, alpha, **kwargs):
    E_te, E_tr, R = get_risk_predictors(gamma, alpha)
    k_crit = binary_search(R,0,n_steps=30, init=1.5, LB=1.0, UB= 2.0)
    ks = np.logspace(np.log10(k_crit), 4, num=100)
    E_tr_inv = get_E_tr_inv(gamma,alpha)
    ks = np.array([E_tr_inv(tau) for tau in np.linspace(0.001,0.98,100)])
    # plt.scatter(df["rreg_E_tr"], df["rreg_E_te"],color = '#1f77b4',label='near-interpol-RR')
    
    
    plt.plot(E_tr(ks),E_te(ks),**kwargs)
    # ks = np.array([E_tr_inv(tau) for tau in np.linspace(0.001,0.98,20)])
    # plt.scatter(E_tr(ks),E_te(ks))
plot_trade_off_curve(gamma=2/3,alpha=2.5, label=r'$\alpha = 1.75$')
plot_trade_off_curve(gamma=2/3,alpha=1.25, label=r'$\alpha = 1.75$')

plt.axhline(y=1, color='k', linestyle=":", label='noise floor')
plt.xlim((-0.05,1.05))


plt.xlabel(r'Sample size $n$')
plt.ylabel(r'Squared norm of params/weights')


# In[ ]:




