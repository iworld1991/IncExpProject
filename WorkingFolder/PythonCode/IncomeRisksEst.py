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


# + {"code_folding": [6, 22, 26, 32, 51, 65, 92, 123]}
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
                 process_para = {'ma_coeffs':np.ones(0),
                                 'sigmas': np.ones((2,100))}
                ):
        self.process_para = process_para
        self.ma_coeffs = process_para['ma_coeffs']
        self.ma_q = len(process_para)
        self.t = t
        self.sigmas = process_para['sigmas']
    
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
    
    def ComputeGenMoments(self):
        ## parameters 
        t = self.t 
        ma_coeffs = self.ma_coeffs
        sigmas = self.sigmas
        p_sigmas = sigmas[0,:]
        t_sigmas = sigmas[1,:]
        ma_q = self.ma_q 
        
        ## generalized moments 
        mean_diff = np.zeros(t) 
        ## varcov is basically the variance covariance of first difference of income of this IMA(q) process
        ## Cov(delta y_t - delta y_{t+k}) forall k for all t
        varcov_diff = np.asmatrix( np.zeros(t**2).reshape([t,t]) )
        
        for i in range(t):
            autocovf_this = p_sigmas[i]**2 + t_sigmas[i]**2 + t_sigmas[i-1]**2
            varcov_diff[i,i] = autocovf_this
            try:
                varcov_diff[i,i+1] = - t_sigmas[i]**2
                varcov_diff[i+1,i] = - t_sigmas[i]**2            
            except:
                pass
        self.GenMoms = {'Mean':mean_diff,
                       'Var':varcov_diff}
        return self.GenMoms
    
    def GetDataMoments(self,
                      data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
    def EstimatePara(self,
                         method = 'TNC',
                         options = {'disp':True}):
        data_moms_dct = self.data_moms_dct
        t = self.t
        
        def ObjFunc(self,
                    para_est = self.process_para):
            an_instance = self.__init__(t = t,
                                        process_para = {'ma_coeffs':para_est[0],
                                'sigmas':para_est[1]}
                                       )
            #a_sim = self.SimulateSeries()
            model_moms_dct = an_instance.ComputeGenMoments() 
            model_moms = np.array([model_moms_dct[key] for key in model_moms_dct.keys()])
            data_moms = np.array([data_moms_dct[key] for key in data_moms_dct.keys()])
            diff = np.linalg.norm(model_moms_dct - data_moms)
            return diff
        
        para_est = minimize(ObjFunc,
                           x0 = None,
                           method = method,
                           options = options)
        
        self.para_est = para_est
        return self.para_est    
        
    def EstimateParabySim(self,
                         method = 'TNC',
                         options = {'disp':True}):
        data_moms_dct = self.data_moms_dct
        process_para_default = self.process_para
        
        def ObjFunc(self,
                    para_est = process_para_default):
            self.process_para = {'ma_coeffs':para_est[0],
                                'sigmas':para_est[1]}
            a_sim = self.SimulateSeries()
            model_moms_sim_dct = self.SimulatedMoments() 
            model_moms = np.array([model_moms_sim_dct[key] for key in model_moms_sim_dct.keys()])
            data_moms = np.array([data_moms_dct[key] for key in data_moms_dct.keys()])
            diff = np.linalg.norm(model_moms_sim - data_moms)
            return diff
        
        para_est = minimize(ObjFunc,
                           x0 = None,
                           method = method,
                           options = options)
        
        self.para_est = para_est
        return self.para_est


# + {"code_folding": [0, 10, 14]}
## debugging test of the data 

t = 50
ma_nosa = np.ones(1)
p_sigmas = np.arange(t)  # sizes of the time-varying permanent volatility 
p_sigmas_rw = np.ones(t) # a special case of time-invariant permanent volatility, random walk 
p_sigmas_draw = np.random.uniform(0,1,t) ## allowing for time-variant shocks 

pt_ratio = 0.33
t_sigmas = pt_ratio * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

dt = IMAProcess(t = t,
               process_para = {'ma_coeffs':ma_nosa,
                              'sigmas':sigmas})
sim_data = dt.SimulateSeries(n_sim = 5000)
sim_moms = dt.SimulatedMoments()

# + {"code_folding": [0]}
## get the computed moments 

comp_moms = dt.ComputeGenMoments()

av_comp = comp_moms['Mean']
cov_var_comp = comp_moms['Var']
var_comp = np.diagonal(cov_var)
autovarb1_comp = np.array([cov_var_comp[i,i+1] for i in range(len(cov_var_comp)-1)]) 

# + {"code_folding": [0]}
## get the simulated moments 
av = sim_moms['Mean']
cov_var = sim_moms['Var']
var = np.diagonal(cov_var)
autovarb1 = np.array([cov_var[i,i+1] for i in range(len(cov_var)-1)]) 

# + {"code_folding": [0]}
## plot simulated moments of first diff 

plt.figure(figsize=((15,4)))

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
plt.legend(loc=0)

# + {"code_folding": []}
## robustness check if the transitory risks is approximately equal to the assigned level

sigma_t_est = np.array(np.sqrt(abs(autovarb1)))
plt.plot(sigma_t_est,label=r'$\widehat sigma_{\theta,t}$')
plt.plot(t_sigmas[1:],label=r'$sigma_{\theta,t}$')
plt.legend(loc=1)

# + {"code_folding": []}
## test of estimator by simulation 

sim_moms_fake = sim_moms.copy()
new_IMA = cp.deepcopy(dt)
new_IMA.GetDataMoments(sim_moms_fake)
new_IMA.EstimatePara()
# -

