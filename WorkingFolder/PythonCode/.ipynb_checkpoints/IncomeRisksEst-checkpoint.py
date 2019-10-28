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


# + {"code_folding": [1, 6, 10, 24, 28, 34, 53, 67, 77, 91, 119, 123, 137, 153, 167, 183, 198, 208, 218]}
## class of integrated moving average process, trend/cycle process allowing for serial correlation transitory shocks
class IMAProcess:
    '''
    inputs
    ------
    t: int, number of periods of the series
    process_para, dict, includes 
       - ma_coeffs: size f q for MA(q),  moving average coeffcients of transitory shocks. q = 0 by default.
       - sigmas:  size of t x 2, draws of permanent and transitory risks from time varying volatility 
    '''
    def __init__(self,
                 t = 100,
                 n_periods = 1,
                 ma_coeffs = np.ones(1),
                 sigmas = np.ones([2,100]),
                ):
        #self.process_para = process_para
        self.ma_coeffs = ma_coeffs
        self.ma_q = self.ma_coeffs.shape[0]
        self.t = t
        self.sigmas =sigmas
        self.n_periods = n_periods
    
    ## auxiliary function for ma cum sum
    def cumshocks(self,
                  shocks,
                  ma_coeffs):
        cum = []
        for i in range(len(shocks)):
            #print(shocks[i])
            #print(sum([ma_coeffs[back]*shocks[i-back] for back in range(len(ma_coeffs))]))
            cum.append(sum([ma_coeffs[back]*shocks[i-back] for back in range(len(ma_coeffs))]))
        return np.array(cum)         
    
    def SimulateSeries(self,
                      n_sim = 100):
        t = self.t 
        ma_coeffs = self.ma_coeffs
        sigmas = self.sigmas
        ma_q = self.ma_q 
                 
        p_draws = np.multiply(np.random.randn(n_sim*t).reshape([n_sim,t]), 
                              np.tile(sigmas[0,:],[n_sim,1]))  # draw permanent shocks
        t_draws = np.multiply(np.random.randn(n_sim*t).reshape([n_sim,t]), 
                              np.tile(sigmas[1,:],[n_sim,1]))  ## draw one-period transitory shocks 
        t_draws_cum = np.array( [self.cumshocks(shocks = t_draws[i,:],
                                                ma_coeffs = ma_coeffs) 
                                 for i in range(n_sim)]
                              )
        series = np.cumsum(p_draws,axis = 1) + t_draws_cum 
        self.simulated = series
        return self.simulated 
       
    def SimulatedMoments(self):
        series = self.simulated 
        
        ## the first difference 
        diff = np.diff(series,axis=1)
        
        ## moments of first diff
        mean_diff = np.mean(diff,axis = 0)
        varcov_diff = np.cov(diff.T)
        
        self.SimMoms = {'Mean':mean_diff,
                       'Var':varcov_diff}
        return self.SimMoms
    
    def TimeAggregate(self,
                      n_periods = None):
        simulated = self.simulated
        t = self.t
        n_periods = self.n_periods
        
        simulated_agg = np.array([np.sum(simulated[:,i-n_periods:i],axis=1) for i in range(n_periods,t+1)]).T
        self.simulated_agg = simulated_agg
        return self.simulated_agg
    
    def SimulateMomentsAgg(self):
        series_agg = self.simulated_agg 
        
        ## the first difference 
        diff = np.diff(series_agg,
                       axis = 1)
        ## moments of first diff
        mean_diff = np.mean(diff,axis = 0)
        varcov_diff = np.cov(diff.T)
        
        self.SimAggMoms = {'Mean':mean_diff,
                           'Var':varcov_diff}
        return self.SimAggMoms
    
    def ComputeGenMoments(self):
        ## parameters 
        t = self.t 
        ma_coeffs = self.ma_coeffs
        sigmas = self.sigmas
        p_sigmas = sigmas[0,:]
        t_sigmas = sigmas[1,:]
        ma_q = self.ma_q 
        
        ## generalized moments 
        mean_diff = np.zeros(t)[1:] 
        ## varcov is basically the variance covariance of first difference of income of this IMA(q) process
        ## Cov(delta y_t - delta y_{t+k}) forall k for all t
        varcov_diff = np.asmatrix( np.zeros((t)**2).reshape([t,t]) )
        
        for i in range(t):
            autocovf_this = p_sigmas[i]**2 + t_sigmas[i]**2 + t_sigmas[i-1]**2
            varcov_diff[i,i] = autocovf_this
            try:
                varcov_diff[i,i+1] = - t_sigmas[i]**2
                varcov_diff[i+1,i] = - t_sigmas[i]**2            
            except:
                pass
        varcov_diff = varcov_diff[1:,1:]
        self.GenMoms = {'Mean':mean_diff,
                       'Var':varcov_diff}
        return self.GenMoms
    
    def GetDataMoments(self,
                      data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
    def ObjFunc(self,
                para):
        data_moms_dct = self.data_moms_dct
        t = self.t
        ma_coeffs,sigmas = para
        self.t = t
        self.ma_coeffs = ma_coeffs
        self.sigmas = sigmas
        model_moms_dct = self.ComputeGenMoments() 
        model_moms = np.array([model_moms_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_dct[key] for key in ['Var']]).flatten()
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
        
    def EstimatePara(self,
                     method = 'CG',
                     bounds = None,
                     para_guess =(1,
                                  np.random.uniform(0,1,100).reshape(2,50)),
                     options = {'disp':True}):
        
        para_est = minimize(self.ObjFunc,
                            x0 = para_guess,
                            method = method,
                            bounds = bounds,
                            options = options)['x']
        
        self.para_est = para_est
        return self.para_est    
    
    def ObjFuncSim(self,
                para_sim):
        data_moms_dct = self.data_moms_dct
        t = self.t
        ma_coeffs,sigmas = para_sim
        self.t = t
        self.ma_coeffs = ma_coeffs
        self.sigmas = sigmas
        model_moms_dct = self.SimulatedMoments() 
        model_moms = np.array([model_moms_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_dct[key] for key in ['Var']]).flatten()
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
        
    def EstimateParabySim(self,
                     method = 'CG',
                     bounds = None,
                     para_guess =(1,
                                  np.random.uniform(0,1,100).reshape(2,50)),
                     options = {'disp':True}):
        
        para_est_sim = minimize(self.ObjFuncSim,
                            x0 = para_guess,
                            method = method,
                            bounds = bounds,
                            options = options)['x']
        
        self.para_est_sim = para_est_sim
        return self.para_est_sim  
    
    def EstimateParabySim2(self,
                         method = 'TNC',
                         options = {'disp':True}):
        data_moms_dct = self.data_moms_dct
        process_para_default = self.process_para
        
        
        para_est = minimize(ObjFunc,
                           x0 = None,
                           method = method,
                           options = options)
        
        self.para_est = para_est
        return self.para_est
    
    def Autocovar(self,
                  step = 1):
        cov_var = self.SimMoms['Var']
        if step >= 0:
            autovar = np.array([cov_var[i,i+step] for i in range(len(cov_var)-1)])
        if step < 0:
            autovar = np.array([cov_var[i+step,i] for i in range(abs(step),len(cov_var)-1)]) 
        self.autovar = autovar
        return self.autovar
    
    def AutocovarComp(self,
                  step = 1):
        cov_var = self.GenMoms['Var']
        if step >= 0:
            autovar = np.array([cov_var[i,i+step] for i in range(len(cov_var)-1)])
        if step < 0:
            autovar = np.array([cov_var[i+step,i] for i in range(abs(step),len(cov_var)-1)]) 
        self.autovarGen = autovar
        return self.autovarGen
    
    def AutocovarAgg(self,
                     step = 0):
        cov_var = self.SimAggMoms['Var']
        if step >=0:
            autovar = np.array([cov_var[i,i+step] for i in range(len(cov_var)-1)]) 
        if step < 0:
            autovar = np.array([cov_var[i,i+step] for i in range(abs(step),len(cov_var)-1)]) 
        self.autovar = autovar
        self.autovaragg = autovar
        return self.autovaragg 

# + {"code_folding": [0]}
## debugging test of the data 

t = 50
ma_nosa = np.array([1])
p_sigmas = np.arange(t)  # sizes of the time-varying permanent volatility 
p_sigmas_rw = np.ones(t) # a special case of time-invariant permanent volatility, random walk 
p_sigmas_draw = np.random.uniform(0,1,t) ## allowing for time-variant shocks 

pt_ratio = 0.33
t_sigmas = pt_ratio * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

dt = IMAProcess(t = t,
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
agg_series = dt.TimeAggregate(n_periods = 2)
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

# + {"code_folding": [0]}
## some fake data moments with alternative parameters

pt_ratio_fake = 0.6
t_sigmas = pt_ratio_fake * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

dt_fake = IMAProcess(t = t,
                     ma_coeffs = ma_nosa,
                     sigmas = sigmas)
sim_data = dt_fake.SimulateSeries(n_sim = 5000)
sim_moms = dt_fake.SimulatedMoments()
# -

# ### Estimation
#
# #### Estimation using computed moments 

# + {"code_folding": []}
## estimation of income risks 

dt_est = cp.deepcopy(dt)
dt_est.GetDataMoments(sim_moms)
# -

para_est = dt_est.EstimatePara(method='CG')


# + {"code_folding": [0]}
## check the estimation and true parameters 

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent shocks')
plt.plot(dt_est.para_est[1][0].T**2,'r-',label='Estimation')
plt.plot(dt_est.sigmas[0]**2,'o',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory shocks')
plt.plot(dt_est.para_est[1][1].T**2,'r-',label='Estimation')
plt.plot(dt_est.sigmas[1]**2,'o',label='Truth')
plt.legend(loc=0)

# + {"code_folding": [], "cell_type": "markdown"}
# #### Estimation using simulated moments 
# -

para_est_sim = dt_est.EstimateParabySim(method='CG')

# + {"code_folding": []}
## check the estimation and true parameters

fig = plt.figure(figsize=([10,4]))

plt.subplot(1,2,1)
plt.title('Permanent shocks')
plt.plot(dt_est.para_est_sim[1][0].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_est.sigmas[0]**2,'o',label='Truth')


plt.subplot(1,2,2)
plt.title('Transitory shocks')
plt.plot(dt_est.para_est_sim[1][1].T**2,'r-',label='Estimation(sim)')
plt.plot(dt_est.sigmas[1]**2,'o',label='Truth')
plt.legend(loc=0)
