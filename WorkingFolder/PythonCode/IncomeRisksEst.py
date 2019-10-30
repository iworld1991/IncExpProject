# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Income Risks Estimation 
#
# This noteobok contains the following
#
#  - Estimation functions of time-varying income risks for an integrated moving average process of income/earnings
#  - It allows for different assumptions about expectations, ranging from rational expectation to alternative assumptions. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy as cp

from IncomeProcess import IMAProcess as ima

# + {"code_folding": [0]}
## debugging test of the data 

t = 66
ma_nosa = np.array([1])
p_sigmas = np.arange(t)  # sizes of the time-varying permanent volatility 
p_sigmas_rw = np.ones(t) # a special case of time-invariant permanent volatility, random walk 
p_sigmas_draw = np.random.uniform(0,1,t) ## allowing for time-variant shocks 

pt_ratio = 0.33
t_sigmas = pt_ratio * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

dt = ima(t = t,
         ma_coeffs = ma_nosa,
         sigmas = sigmas)
sim_data = dt.SimulateSeries(n_sim = 5000)
sim_moms = dt.SimulatedMoments()

# + {"code_folding": [0]}
## get the computed moments 

comp_moms = dt.ComputeGenMoments()

av_comp = comp_moms['Mean']
cov_var_comp = comp_moms['Var']
var_comp = dt.AutocovarComp(step=0) #np.diagonal(cov_var_comp)
autovarb1_comp = dt.AutocovarComp(step=-1)  #np.array([cov_var_comp[i,i+1] for i in range(len(cov_var_comp)-1)]) 

# + {"code_folding": [0]}
## get the simulated moments 
av = sim_moms['Mean']
cov_var = sim_moms['Var']
var = dt.Autocovar(step = 0)   #= np.diagonal(cov_var)
autovarb1 = dt.Autocovar(step = -1) #np.array([cov_var[i,i+1] for i in range(len(cov_var)-1)]) 

# + {"code_folding": [0]}
## plot simulated moments of first diff 

plt.figure(figsize=((20,4)))

plt.subplot(1,4,1)
plt.title(r'$\sigma_{\theta,t},\sigma_{\epsilon,t}$')
plt.plot(p_sigmas_draw,label='sigma_p')
plt.plot(t_sigmas,label='sigma_t')
plt.legend(loc=0)

plt.subplot(1,4,2)
plt.title(r'$\Delta(y_t)$')
plt.plot(av,label='simulated')
plt.plot(av_comp,label='computed')
plt.legend(loc=0)

plt.subplot(1,4,3)
plt.title(r'$Var(\Delta y_t)$')
plt.plot(var,label='simulated')
plt.plot(var_comp,label='computed')
plt.legend(loc=0)

plt.subplot(1,4,4)
plt.title(r'$Cov(\Delta y_t,\Delta y_{t+1})$')
plt.plot(autovarb1,label='simulated')
plt.plot(autovarb1_comp,label='computed')
plt.legend(loc = 0)

# + {"code_folding": []}
## robustness check if the transitory risks is approximately equal to the assigned level

sigma_t_est = np.array(np.sqrt(abs(autovarb1)))
plt.plot(sigma_t_est,'r-',label=r'$\widehat \sigma_{\theta,t}$')
plt.plot(t_sigmas[1:],'b-.',label=r'$\sigma_{\theta,t}$')
plt.legend(loc=1)
# -

# ### Time Aggregation

# + {"code_folding": [0]}
## time aggregation 

sim_data = dt.SimulateSeries(n_sim = 1000)
agg_series = dt.TimeAggregate(n_periods = 3)
agg_series_moms = dt.SimulateMomentsAgg()

# + {"code_folding": [1]}
## difference times degree of time aggregation leads to different autocorrelation
for ns in np.array([2,8]):
    an_instance = cp.deepcopy(dt)
    series = an_instance.SimulateSeries(n_sim =500)
    agg_series = an_instance.TimeAggregate(n_periods = ns)
    agg_series_moms = an_instance.SimulateMomentsAgg()
    var_sim = an_instance.AutocovarAgg(step=0)
    var_b1 = an_instance.AutocovarAgg(step=-1)
    plt.plot(var_b1,label=r'={}'.format(ns))
plt.legend(loc=1)
# -

# ### Estimation

# + {"code_folding": [0]}
## some fake data moments with alternative parameters

pt_ratio_fake = 0.6
t_sigmas = pt_ratio_fake * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

dt_fake = ima(t = t,
              ma_coeffs = ma_nosa,
              sigmas = sigmas)
data_fake = dt_fake.SimulateSeries(n_sim = 5000)
moms_fake = dt_fake.SimulatedMoments()
# -

# #### Estimation using computed moments 

# + {"code_folding": [0]}
## estimation of income risks 

dt_est = cp.deepcopy(dt)
dt_est.GetDataMoments(moms_fake)

para_guess_this = np.ones(2*t  + dt_est.ma_q)  # make sure the length of the parameters are right 

# + {"code_folding": [0]}
para_est = dt_est.EstimatePara(method='CG',
                              para_guess = para_guess_this)


# + {"code_folding": [0]}
## check the estimation and true parameters 

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(dt_est.para_est[1][0].T**2,'r-',label='Estimation')
plt.plot(dt_fake.sigmas[0]**2,'-*',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(dt_est.para_est[1][1].T**2,'r-',label='Estimation')
plt.plot(dt_fake.sigmas[1]**2,'-*',label='Truth')
plt.legend(loc=0)

# + {"code_folding": [], "cell_type": "markdown"}
# #### Estimation using simulated moments 
# -

para_guess_this2 = para_guess_this*0.3
para_est_sim = dt_est.EstimateParabySim(method='Powell',
                                        para_guess = para_guess_this2,
                                        options={'disp':True,
                                                'ftol': 0.000000001})

# + {"code_folding": [0]}
## check the estimation and true parameters

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(dt_est.para_est_sim[1][0][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[0][1:]**2,'-*',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(dt_est.para_est_sim[1][1][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[1][1:]**2,'-*',label='Truth')
plt.legend(loc=0)

# + {"code_folding": [0]}
### reapeating the estimation for many times

n_loop = 10

para_est_sum_sim = (np.array([0]),np.zeros([2,50]))
for i in range(n_loop):
    para_est_this_time = dt_est.EstimateParabySim(method='Powell',
                                                     para_guess = para_guess_this2,
                                                     options = {'disp': True,
                                                               'ftol': 0.0000001})
# -

para_est_av = sum([abs(para_est_sum_sim[2*i+1]) for i in range(1,n_loop+1)] )/n_loop

# + {"code_folding": [0]}
## check the estimation and true parameters

fig = plt.figure(figsize=([14,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(para_est_av[0][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[0][1:]**2,'-*',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(para_est_av[1][1:].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_fake.sigmas[1][1:]**2,'-*',label='Truth')
plt.legend(loc=0)
# -

# #### Estimation using time aggregated data

# + {"code_folding": []}
## get some fake aggregated data moments
moms_agg_fake = dt_fake.TimeAggregate()
moms_agg_dct_fake = dt_fake.SimulateMomentsAgg()

# + {"code_folding": []}
## estimation 
dt_est.GetDataMomentsAgg(moms_agg_dct_fake)
para_est_agg = dt_est.EstimateParaAgg()

# + {"code_folding": [0]}
## check the estimation and true parameters

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(dt_est.para_est_agg[1][0].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[0]**2,'-*',label='Truth')

plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(dt_est.para_est_agg[1][1].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[1]**2,'-*',label='Truth')
plt.legend(loc=0)

# + {"code_folding": [0]}
### reapeating the estimation for many times

n_loop = 20

para_est_sum_agg = (np.array([0]),np.zeros([2,50]))
for i in range(n_loop):
    para_est_this_time = abs(dt_est.EstimateParaAgg(method='CG'))
    para_est_sum_agg = para_est_sum_agg + para_est_this_time
    
para_est_av_agg = para_est_sum_agg/n_loop

# + {"code_folding": []}
## check the estimation and true parameters

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent Risk')
plt.plot(para_est_av_agg[1][0].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[0]**2,'-*',label='Truth')

plt.subplot(1,2,2)
plt.title('Transitory Risk')
plt.plot(para_est_av_agg[1][1].T**2,'r-',label='Estimation(agg)')
plt.plot(dt_fake.sigmas[1]**2,'-*',label='Truth')
plt.legend(loc=0)
