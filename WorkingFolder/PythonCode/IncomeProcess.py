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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root 
import copy as cp


# + {"code_folding": [0, 8]}
def toVec(ma_coeffs,
          sigmas,
          t,
          ma_q):
    assert ma_coeffs.shape == (ma_q,)
    assert sigmas.shape == (2, t)
    return np.hstack([ma_coeffs.flatten(), sigmas.flatten()])

def toPara(vec,
           t,
           ma_q):
    assert len(vec) == 2*t + ma_q
    return vec[:ma_q-1], vec[ma_q:].reshape(2,t)


# + {"code_folding": [0, 6, 10, 24, 28, 34, 55, 69, 78, 82, 95, 119, 128, 156, 160, 164, 180, 200, 214, 231, 251, 279, 297, 307]}
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
                 n_periods = np.array([1]),
                 ma_coeffs = np.ones(1),
                 sigmas = np.ones([2,100]),
                ):
        #self.process_para = process_para
        self.ma_coeffs = ma_coeffs
        self.ma_q = self.ma_coeffs.shape[0]
        self.t = t
        self.sigmas = sigmas
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
                       n_sim = 200):
        t = self.t 
        ma_coeffs = self.ma_coeffs
        sigmas = self.sigmas
        ma_q = self.ma_q 
        np.random.seed(12345)                 
        p_draws = np.multiply(np.random.randn(n_sim*t).reshape([n_sim,t]), 
                              np.tile(sigmas[0,:],[n_sim,1]))  # draw permanent shocks
        np.random.seed(12342)
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
                      n_periods = 1):
        simulated = self.simulated
        t = self.t
        
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
    
##########
## new ####
###########
    def ComputeMomentsAgg(self,
                          n_agg = 1):
        self.sigmas = sigmas
        sigmas_theta = sigmas[0,:]
        sigmas_eps = sigmas[1,:]
        
        n = n_agg 
        t = self.t 
        
        t_truc = t - n 
        
        ## prepare the locations for var-cov matrix 
        var_cov = np.zeros([t_truc,t_truc])
        
        ## prepare a 2n-1 x 1  vector [1,2...n,n-1..1]
        M_vec0 = np.arrange(n-1)+1
        M_vec1 = np.flip(np.arrange(n)+1)  
        M_vec =  np.concatenate((M_vec0,M_vec1))
        
        ## prepare a 2n-2 x 1 vector [-1,-1...,1,1]
        I_vec0 = - np.ones(n-1)
        I_vec1 = np.ones(n-1)
        I_vec =  np.concatenate((I_vec0,I_vec1))
        
        for i in range(t_truc):
            for k in range(n):
                var_cov[i,i+k] = (M_vec[k:]*M_vec[:-k]*sigmas_theta[k:-k]
                                  +I_vec[k:]*I_vec[:-k]*sigmas_eps[k:-k]) # need to check 
                var_cov[i+k,i] = var_cov[i,i+k]
        
        self.Moments_Agg = var_cov
        return self.Moments_Agg
    
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
        
    def GetDataMomentsAgg(self,
                          data_moms_agg_dct):
        self.data_moms_agg_dct = data_moms_agg_dct    
        
    def ObjFunc(self,
                para):
        data_moms_dct = self.data_moms_dct
        t = self.t
        ma_q = self.ma_q
        ma_coeffs,sigmas = toPara(para,
                                  t,
                                  ma_q)
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
                     para_guess = None,
                     options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        
        para_est = minimize(self.ObjFunc,
                            x0 = para_guess,
                            method = method,
                            bounds = bounds,
                            options = options)['x']
        
        self.para_est = toPara(para_est,
                               t,
                               ma_q)
        
        return self.para_est    
    
    def EstimateParaRoot(self,
                         para_guess = None):
        t = self.t
        ma_q = self.ma_q
        
        para_est = root(self.ObjFunc,
                       x0 = para_guess)['x']
        
        self.para_est = toPara(para_est,
                               t,
                               ma_q)
        
        return self.para_est 
    
    def ObjFuncSim(self,
                   para_sim):
        data_moms_dct = self.data_moms_dct
        t = self.t
        ma_q = self.ma_q
        ma_coeffs,sigmas = toPara(para_sim,
                                  t,
                                  ma_q)
        self.ma_coeffs = ma_coeffs
        self.sigmas = sigmas
        model_series_sim = self.SimulateSeries(n_sim = 2000) 
        model_moms_dct = self.SimulatedMoments()  
        model_moms = np.array([model_moms_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_dct[key] for key in ['Var']]).flatten()
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
        
    def EstimateParabySim(self,
                          method = 'CG',
                          bounds = None,
                          para_guess = None,
                          options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        
        para_est_sim = minimize(self.ObjFuncSim,
                                x0 = para_guess,
                                method = method,
                                bounds = bounds,
                                options = options)['x']
        
        self.para_est_sim = toPara(para_est_sim,
                                   t,
                                   ma_q)
        
        return self.para_est_sim    
    
    def ObjFuncAgg(self,
                   para_agg):
        data_moms_agg_dct = self.data_moms_agg_dct
        t = self.t
        ma_q = self.ma_q
        n_periods = self.n_periods
        ma_coeffs,sigmas = toPara(para_agg,
                                  t,
                                  ma_q)
        new_instance = cp.deepcopy(self)
        new_instance.t = t   
        new_instance.ma_coeffs = ma_coeffs
        new_instance.sigmas = sigmas
        model_series_sim = new_instance.SimulateSeries() 
        model_series_agg = new_instance.TimeAggregate(n_periods = n_periods)
        model_moms_agg_dct = new_instance.SimulateMomentsAgg()
        
        model_moms = np.array([model_moms_agg_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_agg_dct[key] for key in ['Var']]).flatten()
        if len(model_moms) > len(data_moms):
            n_burn = len(model_moms) - len(data_moms)
            model_moms = model_moms[n_burn:]
        if len(model_moms) < len(data_moms):
            n_burn = -(len(model_moms) - len(data_moms))
            data_moms = data_moms[n_burn:]
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
    
    def EstimateParaAgg(self,
                        method = 'CG',
                        bounds = None,
                        para_guess = None,
                        options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        para_est_agg = minimize(self.ObjFuncAgg,
                                x0 = para_guess,
                                method = method,
                                bounds = bounds,
                                options = options)['x']
        
        self.para_est_agg = toPara(para_est_agg,
                                   t,
                                   ma_q)
        return self.para_est_agg  
    
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
