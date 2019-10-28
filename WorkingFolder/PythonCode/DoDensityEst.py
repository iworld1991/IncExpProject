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

# ### Density Estimation of Income Risks
#
# - Following Manski et al.(2009)
# - Three cases 
#    - case 1. 3+ intervales with positive probabilities, to be fitted with a generalized beta distribution
#    - case 2. exactly 2 adjacent intervals with positive probabilities, to be fitted with a triangle distribution 
#    - case 3. one interval only, to be fitted with a uniform distribution

from scipy.stats import gamma
from scipy.stats import beta 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import pandas as pd

from DensityEst import SynDensityStat 

# + {"code_folding": []}
### loading probabilistic data  
IndSCE=pd.read_stata('../SurveyData/SCE/IncExpSCEProbIndM.dta')   
# monthly income growth 
# -

IndSCE.head()

## how many observations?
len(IndSCE)

## how many observations have density forecasts
len(IndSCE['Q24_bin10'].dropna())

IndSCE['Q24_bin3'].dropna()

# + {"code_folding": []}
## survey-specific parameters 
nobs=len(IndSCE)
SCE_bins=np.array([-20,-12,-8,-4,-2,0,2,4,8,12,20])
print("There are "+str(len(SCE_bins)-1)+" bins in SCE")

# + {"code_folding": [12]}
##############################################
### attention: the estimation happens here!!!!!
###################################################


## creating positions 
index  = IndSCE.index
columns=['IncMean','IncVar']
IndSCE_moment_est = pd.DataFrame(index=index,
                                 columns=columns)

## Invoking the estimation
for i in range(nobs):
    print(i)
    ## take the probabilities (flip to the right order, normalized to 0-1)
    Inc = np.flip(np.array([IndSCE.iloc[i,:]['Q24_bin'+str(n)]/100 for n in range(1,11)]))
    print(Inc)
    if not np.isnan(Inc).any():
        stats_est = SynDensityStat(SCE_bins,Inc)
        if len(stats_est)>0:
            IndSCE_moment_est['IncMean'][i] = stats_est['mean']
            print(stats_est['mean'])
            IndSCE_moment_est['IncVar'][i] = stats_est['variance']
            print(stats_est['variance'])
# -

### exporting moments estimates to pkl
IndSCE_est = pd.concat([IndSCE,IndSCE_moment_est], join='inner', axis=1)
IndSCE_est.to_pickle("./IndSCEDstIndM.pkl")
IndSCE_pk = pd.read_pickle('./IndSCEDstIndM.pkl')

IndSCE_pk['IncMean']=pd.to_numeric(IndSCE_pk['IncMean'],errors='coerce')   # income growth from y-1 to y 
IndSCE_pk['IncVar']=pd.to_numeric(IndSCE_pk['IncVar'],errors='coerce')   


IndSCE_pk.tail()

columns_keep = ['date','year','month','userid','tenure','IncMean','IncVar']
IndSCE_pk_new = IndSCE_pk[columns_keep]
IndSCE_pk_new.to_stata('../SurveyData/SCE/IncExpSCEDstIndM.dta')

# + {"code_folding": []}
### Robustness checks: focus on big negative mean estimates 
sim_bins_data = SCE_bins
print(str(sum(IndSCE_pk['IncMean']<-6))+' abnormals')
ct=0
figure=plt.plot()
for id in IndSCE_pk.index[IndSCE_pk['IncMean']<-8]:
    print(id)
    print(IndSCE_pk['IncMean'][id])
    sim_probs_data= np.flip(np.array([IndSCE['Q24_bin'+str(n)][id]/100 for n in range(1,11)]))
    plt.bar(sim_bins_data[1:],sim_probs_data)
    print(sim_probs_data)
    stats_est=SynDensityStat(SCE_bins,sim_probs_data)
    print(stats_est['mean'])
