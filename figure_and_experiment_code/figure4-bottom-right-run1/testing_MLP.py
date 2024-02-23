#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

from ruamel.yaml import YAML
# Create YAML object
yaml = YAML()

# Open the file for reading
with open('config.yaml', 'r') as file:
    # Load the content of the file
    config = yaml.load(file)


# In[84]:


hls = eval(config['hidden_layer_sizes'])
hls
n_trs = eval(config['n_trs'])
alpha = config['alpha']
gamma = eval(config['gamma'])
tau = config['tau']
n_te = config['n_te']
max_iter = config['max_iter']
replicate = config['replicate']


# In[85]:



def run_MLP(X_tr,y_tr,X_te,y_te,E_tr):
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

    
    return {"mlp_norm":total_norms[-1], "mlp_E_te": mse_losses_te[-1], "mlp_converged": False}


def run_single_experiment(gamma, alpha, n_tr, n_te, E_tr, beta_scale = 10):
    print("running exp")
    p = int(n_tr/gamma) # data dimension
    
    idx = np.arange(1,p+1) # feature indices

    pop_evs = idx**(-alpha) # population level eigenvalues

    X_tr = np.multiply(np.sqrt(pop_evs[:,None]), np.random.normal(size= (p, n_tr)) )
    X_te = np.multiply(np.sqrt(pop_evs[:,None]), np.random.normal(size= (p, n_te)) )

    beta_true = np.sqrt(beta_scale)*np.random.normal(size= (p,1))/np.sqrt(p)

    y_tr = X_tr.T@beta_true + np.random.normal(size= (n_tr,1))
    y_te = X_te.T@beta_true + np.random.normal(size= (n_te,1))

    mlp_results = run_MLP(X_tr,y_tr,X_te,y_te,E_tr)
    mlp_results['n_tr'] = n_tr
    mlp_results['tau'] = E_tr
    return mlp_results


# In[86]:


results = [run_single_experiment(alpha,gamma, n_tr=n_tr,n_te = n_te, E_tr= tau) for n_tr in n_trs for _ in range(replicate)]


# In[92]:


df = pd.DataFrame(results)
df.to_csv('outputs/results.csv')


# In[91]:


df


# In[90]:


plt.scatter(df['n_tr'], df['mlp_norm'])

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Sample size $n$')
plt.ylabel(r'Squared norm of params/weights')


# In[ ]:




