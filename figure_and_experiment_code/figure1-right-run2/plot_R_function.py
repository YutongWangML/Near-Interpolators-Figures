#!/usr/bin/env python
# coding: utf-8

# In[8]:


from test_error_analytic_form import get_risk_predictors, k_grid
from binary_search import binary_search,root_search
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
import matplotlib.patheffects as pe

gamma = 0.5
alpha = 1.5
colors = ['#1f77b4', '#419ede']
def plot_R_function(gamma, alpha, **kwargs):
    _, _, R = get_risk_predictors(gamma, alpha)
    ks = np.linspace(0, 3, num=100)
    # plt.scatter(df["rreg_E_tr"], df["rreg_E_te"],color = '#1f77b4',label='near-interpol-RR')
    

    root = root_search(R)
    plt.plot(ks,R(ks),**kwargs)
    plt.axvline(x = root, color='gray', linestyle='--')


# In[12]:


plot_R_function(gamma=0.5,alpha=1.75, label=r'$\gamma = 0.5, \, \alpha = 1.75$')
plot_R_function(gamma=0.5,alpha=1.1, label=r'$\gamma = 0.5, \, \alpha = 1.1$')
plot_R_function(gamma=0.3,alpha=1.1, label=r'$\gamma = 0.3, \, \alpha = 1.1$')
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel(r'$k$')
plt.ylabel(r'$r$')
plt.title(r'$\mathcal{R}(k)$')
plt.legend()
plt.savefig('outputs/R_func.svg')
plt.savefig('outputs/R_func.png')


# In[27]:


def plot_E_tr_function(gamma, alpha, **kwargs):
    _, E_tr, R = get_risk_predictors(gamma, alpha)
    # plt.scatter(df["rreg_E_tr"], df["rreg_E_te"],color = '#1f77b4',label='near-interpol-RR')
    

    root = root_search(R)
    ks = np.linspace(root, root+10, num=100)
    plt.plot(ks,E_tr(ks),**kwargs)
    plt.axvline(x = root, color='gray', linestyle='--')


# In[28]:


plot_E_tr_function(gamma=0.5,alpha=1.75, label=r'$\gamma = 0.5, \, \alpha = 1.75$')
plot_E_tr_function(gamma=0.5,alpha=1.1, label=r'$\gamma = 0.5, \, \alpha = 1.1$')
plot_E_tr_function(gamma=0.3,alpha=1.1, label=r'$\gamma = 0.3, \, \alpha = 1.1$')
plt.axhline(y=0, color='k', linestyle=':')
plt.xlabel(r'$k$')
plt.ylabel(r'$\mathcal{E}_{\mathtt{train}}(k)$')
plt.title(r'$\mathcal{E}_{\mathtt{train}}(k)$')
plt.legend()
plt.savefig('outputs/E_tr_func.svg')
plt.savefig('outputs/E_tr_func.png')


# In[ ]:




