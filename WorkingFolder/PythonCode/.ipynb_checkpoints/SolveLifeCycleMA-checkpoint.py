# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Solve a life-cycle consumption/saving problem 
#
# - This notebook reproduces the life cycle consumption model by Gourinchas and Parker 2002 
#
#   - CRRA utility 
#   - No bequest motive
#   - During work: labor income risk: permanent + transitory/unemployment 
#   - During retirement: no risk
#
#

import numpy as np
import pandas as pd
from quantecon.optimize import brent_max, brentq
from interpolation import interp, mlinterp
from scipy import interpolate
import numba as nb
from numba import njit, float64, int64, boolean,jitclass
import matplotlib as mp
import matplotlib.pyplot as plt
# %matplotlib inline
from quantecon import MarkovChain
import quantecon as qe 
from mpl_toolkits.mplot3d import Axes3D

# + code_folding=[0]
## figures configurations

mp.rc('xtick', labelsize=14) 
mp.rc('ytick', labelsize=14) 

fontsize = 14
legendsize = 12
# -

# ## The Model Class and Solver

# + code_folding=[]
lc_data = [
    ('ρ', float64),              # utility parameter CRRA
    ('β', float64),              # discount factor
    ('R',float64),               # Nominal interest rate factor 
    ('P', float64[:, :]),        # transition probs for z_t, a persistent (macro) state  x
    ('z_val', float64[:]),       # values of z, grid values for the continous (macro) persistent state    x
    #('a_s', float64),           # preference volatility        x 
    #('sigma_s', float64),       # loading of macro state to preference    x
    ('sigma_n', float64),        # permanent shock volatility              x
    ('ϕ',float64),               # MA(1) coefficient, or essentially the autocorrelation coef of non-persmanent income
    ('sigma_ϵ', float64),        # transitory shock volatility
    ('U',float64),               # the probability of being unemployed    * 
    ('b_y', float64),            # loading of macro state to income        x 
    ('s_grid', float64[:]),      # Grid over savings
    ('n_shk_draws', float64[:]), ## Draws of permanent income shock 
    ('ϵ_shk_draws', float64[:]), # Draws of MA/transitory income shocks 
    ('ue_shk_draws',boolean[:]), # Draws of unemployment shock 
    #('ζ_draws', float64[:])     # Draws of preference shock ζ for MC
    ('T',int64),                 # years of work                          *   
    ('L',int64),                 # years of life                          * 
    ('G',float64)               # growth rate of permanent income    * 
]


# + code_folding=[7]
@jitclass(lc_data)
class LifeCycle:
    """
    A class that stores primitives for the life cycle consumption
    problem.
    """

    def __init__(self,
                 ρ = 2,
                 β = 0.96,
                 P = np.array([(0.9, 0.1),
                             (0.1, 0.9)]),
                 z_val = np.array([1.0,2.0]), 
                 sigma_n = 0.15,     ## size of permanent income shocks
                 #a_s = 0.02,     ## size of the taste shock  
                 #b_s = 0.0,       ## coefficient of pandemic state on taste 
                 ϕ = 0.1,           ## MA(1) coefficient of non-permanent inocme shocks
                 sigma_ϵ = 0.2,     ## size of transitory income risks
                 U = 0.0,           ## unemployment risk probability (0-1)
                 b_y = 0.0,         ## loading of macro state on income  
                 R = 1.03,           ## interest factor 
                 T = 40,             ## work age, from 25 to 65
                 L = 60,             ## life length 85
                 G = 1.0,            ## growth rate of permanent income 
                 shock_draw_size = 50,
                 grid_max = 2.5,
                 grid_size = 50,
                 seed = 1234):

        np.random.seed(seed)  # arbitrary seed

        self.ρ, self.β = ρ, β
        self.R = R 
        self.P, self.z_val = P, z_val
        self.G = G
        self.T,self.L = T,L
        
        self.sigma_u= sigma_u, 
        self.sigma_n = sigma_n
        self.ϕ = ϕ
        self.sigma_ϵ = sigma_ϵ
        self.b_y = b_y
        
        self.n_shk_draws = sigma_n*np.random.randn(shock_draw_size)-sigma_n**2/2
        self.ϵ_shk_draws = sigma_ϵ*np.random.randn(shock_draw_size)-sigma_ϵ**2/2
        self.ue_shk_draws = np.random.uniform(0,1,shock_draw_size)<U

        self.s_grid = np.exp(np.linspace(np.log(1e-6), np.log(grid_max), grid_size))
        lb_sigma_ϵ = -sigma_ϵ**2/2-2*\sigma_ϵ
        ub_sigma_ϵ = -sigma_ϵ**2/2+2*\sigma_ϵ
        self.ϵ_grid = np.linspace(lb_sigma_ϵ,ub_sigma_ϵ,grid_size)
        
        # This creates an unevenly spaced grids where grids are more dense in low values
        
        # Test stability assuming {R_t} is IID and adopts the lognormal
        # specification given below.  The test is then β E R_t < 1.
        #ER = np.exp(b_r + a_r**2 / 2)
        assert β * R < 1, "Stability condition failed."

    # Marginal utility
    def u_prime(self, c):
        return c**(-self.ρ)

    # Inverse of marginal utility
    def u_prime_inv(self, c):
        return c**(-1/self.ρ)

    #def ϕ(self, z, ζ):
    #    ## preference 
    #    return np.exp(self.sigma_s * ζ + (z*self.b_s))

    def Y(self, z, u_shk):
        ## u_shk is the MA of past shocks 
        ## income 
        return np.exp(u_shk + (z * self.b_y))
    
    def Γ(self,n_shk):
        return np.exp(n_shk)


# + code_folding=[4]
@njit
def K(aϵ_in, σ_in, lc):
    """
    The Coleman--Reffett operator for the life-cycle consumption problem,
    using the endogenous grid method.

        * lc is an instance of life cycle model
        * a_in[i, z] is an asset grid
        * σ_in[i, z] is consumption at a_in[i, z]
    """

    # Simplify names
    u_prime, u_prime_inv = lc.u_prime, lc.u_prime_inv
    R, ρ, P, β = lc.R, lc.ρ, lc.P, lc.β
    z_val = lc.z_val
    s_grid,ϵ_grid = lc.s_grid,lc.ϵ_grid
    n_shk_draws, ϵ_shk_draws, ue_shk_draws= lc.n_shk_draws, lc.ϵ_shk_draws, lc.ue_shk_draws
    Y = lc.Y
    ####################
    ρ = lc.ρ
    Γ = lc.Γ
    G = lc.G
    ϕ = lc.ϕ
    ###################
    
    n = len(P)

    # Create consumption function by linear interpolation
    ########################################################
    σ = lambda a,ϵ,z: mlinterp((aϵ_in[:,0,z],a_in[0,:,z]),σ_in[:,:,z], (a,ϵ)) 
    ########## need to replace with multiinterp 

    # Allocate memory
    σ_out = np.empty_like(σ_in)  ## grid_size_s X grid_size_ϵ X grid_size_z

    # Obtain c_i at each s_i, z, store in σ_out[i, z], computing
    # the expectation term by Monte Carlo
    for i, s in enumerate(s_grid):
        for j, ϵ in enumerate(ϵ_grid):
            for z in range(n):
                # Compute expectation
                Ez = 0.0
                for z_hat in range(n):
                    z_val_hat = z_val[z_hat]
                    for ϵ_shk in lc.ϵ_shk_draws:
                        for ue_shk in lc.ue_shk_draws:
                            for n_shk in lc.n_shk_draws:
                                Γ_hat = Γ(n_shk) 
                                u_shk = ϕ*ϵ+ϵ_shk
                                Y_hat = Y(z_val_hat,u_shk)*(1-ue_shk) ## conditional employed 
                                c_hat = σ(R/(G*Γ_hat) * s + Y_hat,ϵ_shk,z_hat)
                                utility = (G*Γ_hat)**(1-ρ)*u_prime(c_hat)
                                Ez += utility * P[z, z_hat]
                Ez = Ez / (len(n_shk_draws)*len(ϵ_shk_draws)*len(ue_shk_draws))
                σ_out[i, j, z] =  u_prime_inv(β * R* Ez)

    # Calculate endogenous asset grid
    aϵ_out = np.empty_like(σ_out)
    
    for j,ϵ in eneumerate(ϵ_grid):
        for z in range(n):
            aϵ_out[:,j,z] = s_grid + σ_out[:,j,z]

    # Fixing a consumption-asset pair at (0, 0) improves interpolation
    σ_out[0, ,:] = 0.0
    a_out[0, ,:] = 0.0

    return aϵ_out, σ_out


# + code_folding=[0, 22]
## need to modify 
def solve_model_backward_iter(model,        # Class with model information
                              a_vec,        # Initial condition for assets
                              σ_vec,        # Initial condition for consumption
                              #tol=1e-6,
                              #max_iter=2000,
                              #verbose=True,
                              #print_skip=50
                             ):

    # Set up loop
    #i = 0
    #error = tol + 1

    ## memories for life-cycle solutions 
    n_grids = len(σ_vec)
    n_z = len(model.P)                       
    as_new =  np.empty((model.T,n_grids,n_z),dtype = np.float64)
    σs_new =  np.empty((model.T,n_grids,n_z),dtype = np.float64)
    
    as_new[0,:,:] = a_vec
    σs_new[0,:,:] = σ_vec
    
    for i in range(model.T-1):
        print(f"at work age of "+str(model.T-i))
        a_vec_next, σ_vec_next = as_new[i,:,:],σs_new[i,:,:]
        a_new, σ_new = K(a_vec_next, σ_vec_next, model)
        as_new[i+1,:,:] = a_new
        σs_new[i+1,:,:] = σ_new
    
    #while i < max_iter and error > tol:
    #    a_new, σ_new = K(a_vec, σ_vec, model)
    #    error = np.max(np.abs(σ_vec - σ_new))
    #    i += 1
    #    if verbose and i % print_skip == 0:
    #        print(f"Error at iteration {i} is {error}.")
    #    a_vec, σ_vec = np.copy(a_new), np.copy(σ_new)

    #if i == max_iter:
    #    print("Failed to converge!")

    #if verbose and i < max_iter:
    #    print(f"\nConverged in {i} iterations.")

    return as_new, σs_new


# + code_folding=[0]
def policyfunc(lc,
               a_star,
               σ_star,
               discrete = True):
    """
     * ifp is an instance of IFP
        * a_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
    if discrete==True:
        # Create consumption function by linear interpolation
        σ =  lambda a, z_idx: interp(a_star[:, z_idx], σ_star[:, z_idx], a) 
    else:
        # get z_grid 
        z_val = lc.z_val 

        # Create consumption function by linear interpolation
        a = a_star[:,0]                                ## aseet grid 
        σ =  interpolate.interp2d(a, z_val, σ_star.T) 
    
    return σ


# -

# ## Solve the model for some consumption from the last period 

# + code_folding=[0, 2]
## this is the retirement consumption policy 

def policyPF(β,
             ρ,
             R,
             T,
             L):
    c_growth = β**(1/ρ)*R**(1/ρ-1)
    return (1-c_growth)/(1-c_growth**(L-T))


# + code_folding=[0]
## intialize 

lc = LifeCycle()

# Initial the retirement consumption policy of σ = consume all assets

#mpc_ret = policyPF(lc.β,
#                   lc.ρ,
#                   lc.R,
#                   lc.T,
#                   lc.L) 
#ratio = mpc_ret/(1-mpc_ret)

k = len(lc.s_grid)
n = len(lc.P)
σ_init = np.empty((k, n))
a_init = np.empty((k, n))

for z in range(n):
    σ_init[:, z] = 2*lc.s_grid
    a_init[:,z] =  2*lc.s_grid
# -

plt.title('The consumption in the last period')
plt.plot(σ_init[:,1],a_init[:,1])

# + code_folding=[]
#print('The MPC out of cash in hand at the retirement is '+ str(mpc_ret))

# + code_folding=[0]
## Set quarterly parameters 

lc.ρ = 0.5
lc.R = 1.03
lc.β = 0.96

lc.sigma_n = np.sqrt(0.02) # permanent 
lc.sigma_u = np.sqrt(0.04) # transitory 

# + code_folding=[0]
## shut down the macro state 

lc.b_y = 0.00

# + code_folding=[0]
as_star, σs_star = solve_model_backward_iter(lc,
                                             a_init, 
                                             σ_init)

# + [markdown] code_folding=[]
# ### Plot interpolated policy functions

# + code_folding=[3, 5]
ages  = [31,33,35,37,39]

fig = plt.plot()
for age in ages:
    i = lc.T-age
    plt.plot(as_star[i,:,0],
             σs_star[i,:,0],
             label = str(age))
#plt.plot(as_star[0,:,0],as_star[0,:,0],'-')
plt.legend(loc=1)

# + code_folding=[4]
## interpolate consumption function on continuous z grid 

σs_list = []

for i in range(lc.T):
    this_σ= policyfunc(lc,
                   as_star[i,:,:],
                   σs_star[i,:,:],
                   discrete = False)
    σs_list.append(this_σ)
    

# + code_folding=[0]
## plot contour for policy function 

a_grid = np.linspace(0.00001,5,20)
z_grid = np.linspace(0,8,20)
aa,zz = np.meshgrid(a_grid,z_grid)

σ_this = σs_list[3]

c_stars = σ_this(a_grid,z_grid)

cp = plt.contourf(aa, zz,c_stars)
plt.title(r'$c$')
plt.xlabel('asset')
plt.ylabel('another state')

# + code_folding=[0]
## plot 3d consumption function 


x,y,z =σs_star

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'red')
plt.savefig("demo.png")

# + code_folding=[0]
## plot 3d functions 
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dem3d=ax.plot_surface(xv,yv,dem_100,cmap='afmhot', linewidth=0)
ax.set_title('consumption function over life cycle')
ax.set_zlabel('wealth')
plt.show()
# -

# ### Adding a macro Markov/persistent state 

## initialize another 
lc_ag = LifeCycle()

# + code_folding=[0]
## tauchenize an ar1

ρ, σ = (0.98,0.18)
constant = 0.13  

mc = qe.markov.approximation.tauchen(ρ, σ, b=constant, m=3, n=7)
z_ss_av = constant/(1-ρ)
z_ss_sd = σ*np.sqrt(1/(1-ρ**2))

## feed the model with a markov matrix of macro state 
lc_ag.z_val, lc_ag.P = mc.state_values, mc.P

## set the macro state loading to be positive
lc_ag.b_y = 0.1

# + code_folding=[0]
## initialize policies 

k = len(lc_ag.s_grid)
n = len(lc_ag.P)
σ_init = np.empty((k, n))
a_init = np.empty((k, n))

for z in range(n):
    σ_init[:, z] = 2*lc_ag.s_grid
    a_init[:,z] =  2*lc_ag.s_grid

# + code_folding=[0]
as_star_ag, σs_star_ag = solve_model_backward_iter(lc_ag,
                                                   a_init,
                                                   σ_init)

# + code_folding=[]
## interpolate consumption function on continuous z grid 

σs_ag_list = []

for i in range(lc_ag.T):
    this_σ= policyfunc(lc_ag,
                   as_star_ag[i,:,:],
                   σs_star_ag[i,:,:],
                   discrete = False)
    σs_ag_list.append(this_σ)

# + code_folding=[]
## plot contour for policy function 

a_grid = np.linspace(0.00001,5,20)
z_grid = np.linspace(0,8,20)
aa,zz = np.meshgrid(a_grid,z_grid)

σ_this = σs_ag_list[15]

c_stars = σ_this(a_grid,z_grid)

cp = plt.contourf(aa, zz,c_stars)
plt.title(r'$c$')
plt.xlabel('asset')
plt.ylabel('macro state')


# -

# ## Simulate a cross history 

# + code_folding=[1, 28, 48, 78, 100]
#@njit
def simulate_time_series(lc, σ, z_idx_seq, p_income,T=400):
    """
    Simulates a time series of length T for assets/consumptions, given optimal
    consumption/demand functions.
    * z_seq is a time path for {Z_t} recorded by index, instead of its numeric value

    """
    
    # Simulate the asset path
    a = np.zeros(T)+1e-4
    c  = np.empty_like(a)
    #c1 = np.empty_like(a)
    #c2 = np.empty_like(a)
    
    ## simulate histories
    ζ_sim = np.random.randn(T)
    η_sim = np.random.randn(T)
    
    
    R = lc.R
    z_val = lc.z_val ## values of the state 
    
    
    ## permanent income shocks
    
    Γs = p_income[1:]/p_income[:-1] 
    
    for t in range(T):
        z_idx = z_idx_seq[t]
        z = z_val[z_idx]    
        S = lc.ϕ(z,ζ_sim[t])
        Y = lc.Y(z, η_sim[t])
        c[t] = σ(a[t], z_idx)
        #c1[t],c2[t] = allocate(c[t], S = S) 
        #if t<T-1:
        #    a[t+1] = R/Γs[t] * (a[t] - c1[t]*p_vec[0]-c2[t]*p_vec[1]) + Y
        if t<T-1:
            a[t+1] = R/Γs[t] * (a[t] - c[t]) + Y
        
    ## multiply permanent income level 
    #c = c*p_income
    #c1 =c1*p_income
    #c2 = c2*p_income
    #a = a * p_income 
    
    return a,c

def simulate_time_series_new(lc, σ, z_seq, p_income, T=400):
    """
    Simulates a time series of length T for assets/consumptions, given optimal
    consumption/demand functions.

        * ifp is an instance of IFP
        * a_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid
        * z_seq is a time path for {Z_t} recorded by its numeric value (different from the previous function)

    """
    
    # Simulate the asset path
    a = np.zeros(T)+1e-4
    c = np.empty_like(a)
    #c1 = np.empty_like(a)
    #c2 = np.empty_like(a)
    
    ## simulate histories
    ζ_sim = np.random.randn(T)
    η_sim = np.random.randn(T)
    
    
    R = lc.R
    #z_val = ifp.z_val ## values of the state 
    
    ## permanent income shocks
    
    Γs = p_income[1:]/p_income[:-1] 
    
    for t in range(T):
        z = z_seq[t] ## z values
        S = lc.ϕ(z,ζ_sim[t])
        Y = lc.Y(z, η_sim[t])
        c[t] = σ(a[t], z)
        #c1[t],c2[t] = allocate(c[t], S = S) 
        #if t<T-1:
        #    a[t+1] = R/Γs[t] * (a[t] - c1[t]*p_vec[0]-c2[t]*p_vec[1]) + Y
        if t<T-1:
            a[t+1] = R/Γs[t] * (a[t] - c[t]) + Y
        
    ## multiply permanent income level 
    #c = c*p_income
    #c1 =c1*p_income
    #c2 = c2*p_income
    #a = a * p_income 
    
    return a,c

## now, we simulate the time-series of a cross-sectional matrix of N agents 

#@njit
def simulate_distribution(lc, 
                          a_star, 
                          p_vec, 
                          σ_star,
                          z_mat, 
                          p_income_mat,
                          N = 3000, 
                          T = 400,
                          discrete = True):
    N_z, T_z = z_mat.shape
    
    assert N_z>=N and T_z >=T, 'history of the markov states are smaller than the simulated matrix'
    
    
    z_mat = z_mat[0:N,0:T]
    ## z_mat is a N_sim x T sized matrix that takes the simulated Markov states 
    a_mat = np.empty((N,T))
    c_mat = np.empty((N,T))
    #c1_mat = np.empty((N,T))
    #c2_mat = np.empty((N,T))
    
    ## get the policy function
    
    if discrete ==True:
        σ = policyfunc(lc,
                       a_star,
                       σ_star,
                       discrete = True)  ## interpolate for discrete z index 
        for i in range (N):
            a_mat[i,:],c_mat[i,:] = simulate_time_series(lc,
                                                         σ,
                                                         z_mat[i,:],
                                                         p_income_mat[i,:],
                                                         T = T)
    else:
        σ = policyfunc(lc,
                       a_star,
                       σ_star,
                       discrete = False) ## interpolate for continous z value 
        for i in range (N):
            a_mat[i,:],c_mat[i,:]= simulate_time_series_new(lc,
                                                            σ,
                                                            z_mat[i,:],
                                                            p_income_mat[i,:],
                                                            T = T)
            
    ## multiply permanent income level 
    #c_mat= np.multiply(c_mat,p_income_mat)
    #c1_mat = np.multiply(c1_mat,p_income_mat)
    #c2_mat = np.multiply(c2_mat,p_income_mat)
    #a_mat = np.multiply(a_mat,p_income_mat) 

    return a_mat,c_mat

# + code_folding=[]
## simulate a Markov sequence 

mc = MarkovChain(lc.P)

### Simulate history of Idiosyncratic Z states 
#### (For Z to be aggregate state. We can directly copy Z for different agents) 

## number of agents 

N = 1000
T = 25        ## simulated history of time period

z_idx_ts = mc.simulate(T, random_state=13274)
z_idx_mat = np.tile(z_idx_ts,(N,1))


# + code_folding=[3]
## simulate a permanent income distributions 

@njit
def PerIncSimulate(T,
               sigma,
               init = 0.001):
    pshk_draws = sigma*np.random.randn(T)-sigma**2/2
    log_p_inc = np.empty(T)
    log_p_inc[0] = init
    for t in range(T-1):
        log_p_inc[t+1] = log_p_inc[t]+ pshk_draws[t+1]
    p_income = np.exp(log_p_inc)
    return p_income

## simulate histories of permanent income 

p_income_mat = np.empty((N,T))

for n in range(N):
    p_income_mat[n,:] = PerIncSimulate(T,
                                       sigma = lc.sigma_n,
                                       init = 0.0001)

# + code_folding=[]
## Simulate the distribution of consumption/asset (perfect understanding)

p_vec = (1,1) 
a_dist,c_dist = simulate_distribution(lc,
                                      a_bf_star,
                                      p_vec,
                                      σ_bf_star,
                                      z_idx_mat,
                                      p_income_mat,
                                      N = N,
                                      T = T,
                                      discrete = True)

# + code_folding=[]
## aggregate history 

co_mat = np.multiply(c_dist,p_income_mat)  ## non-normalized consumption
lco_mat = np.log(co_mat)
lco_av = np.mean(lco_mat,axis = 0)

#p_av =  np.mean(p_income_mat,axis = 0)  
#lp_av = np.log(p_av)
lp_income_mat = np.log(p_income_mat)   ## permanent income level 
lp_av = np.mean(lp_income_mat,axis = 0)

#c_av = np.mean(c_dist,axis=0)
#lc_av = np.log(c_av)
lc_mat = np.log(c_dist)             ## normalized consumption
lc_av = np.mean(lc_mat,axis = 0) 

lc_sd = np.sqrt(np.diag(np.cov(lc_mat.T)))
# -

plt.title('average log consumption (normalized)')
plt.plot(lc_av[1:],label = r'$\widebar{ln(c/o)}$')
plt.legend(loc=2)

plt.title('average log consumption (non-normalized)')
plt.plot(lco_av[1:],label = r'$\widebar{ln(c)}$')
plt.legend(loc=2)

plt.title('standard deviation of log consumption (normalized)')
plt.plot(lc_sd[1:],label = r'$std(ln(c/o)$')
plt.legend(loc=2)

plt.title('average log permanent income')
plt.plot(lp_av[1:],label = r'$\widebar{ln(o)}$')
plt.legend(loc=2)

# + code_folding=[0]
## get lorenz curve of the consumption inequality 

C_model = c_dist[:,-1]
C1_model = c1_dist[:,-1]
C2_model = c2_dist[:,-1]

## multiply by permanent income 
CO_model = np.multiply(c_dist[:,-1],p_income_mat[:,-1])
CO1_model = np.multiply(c1_dist[:,-1],p_income_mat[:,-1])
CO2_model = np.multiply(c2_dist[:,-1],p_income_mat[:,-1])

fc_m_vals, lc_m_vals = qe.lorenz_curve(CO_model)
fc1_m_vals, lc1_m_vals = qe.lorenz_curve(CO1_model)
fc2_m_vals, lc2_m_vals = qe.lorenz_curve(CO2_model)


fig, axs = plt.subplots(1,
                        3,
                        figsize=(13,4))

## total consumption 
axs[0].plot(fc_vals, lc_vals, 'r-.',label='data')
axs[0].plot(fc_m_vals, lc_m_vals, 'b-',label='model')
axs[0].plot(fc_vals, fc_vals, 'k--',label='equality')
axs[0].legend(fontsize=legendsize)
axs[0].set_title(r'$c$',fontsize=fontsize)
#plt.xlim([0,1])
#plt.ylim([0,1])

## conctact consumption 

## total consumption 
axs[1].plot(fc1_vals, lc1_vals, 'r-.',label='data')
axs[1].plot(fc1_m_vals, lc1_m_vals, 'b-',label='model')
axs[1].plot(fc1_vals, fc1_vals, 'k--',label='equality')
axs[1].legend(fontsize=legendsize)
axs[1].set_title(r'$c_c$',fontsize=fontsize)
#plt.xlim([0,1])
#plt.ylim([0,1])


## total consumption 
axs[2].plot(fc2_vals, lc2_vals, 'r-.',label='data')
axs[2].plot(fc2_m_vals, lc1_m_vals, 'b-',label='model')
axs[2].plot(fc2_vals, fc2_vals, 'k--',label='equality')
axs[2].legend(fontsize=legendsize)
axs[2].set_title(r'$c_n$',fontsize=fontsize)
plt.xlim([0,1])
plt.ylim([0,1])

plt.show()
fig.savefig('../graph/model/lorenz_c.jpg')
# -

# ## Then, solve the model with the pandemic 
#
#

ifp.b_y = -0.1
ifp.b_s = -0.2

# + code_folding=[0]
## Pandemic Markov 

## feed a markov tanchened from ar1
## these parameters are estimated from Covid19 cases per capita of all U.S. counties during the pandemic 

ρ, σ = (0.978,0.18)
constant = 0.13  
mc = qe.markov.approximation.tauchen(ρ, σ, b=constant, m=3, n=7)
z_ss_av = constant/(1-ρ)
z_ss_sd = σ*np.sqrt(1/(1-ρ**2))

## feed ifp with a markov matrix 
ifp.z_val, ifp.P = mc.state_values, mc.P

## some initial guesses 
k = len(ifp.s_grid)
n = len(ifp.P)
σ_init = np.empty((k, n))
for z in range(n):
    σ_init[:, z] = ifp.s_grid
a_init = np.copy(σ_init)
# -

a_star, σ_star = solve_model_time_iter(ifp,a_init, σ_init)

# + code_folding=[]
## interpolate consumption function on continuous z grid 
σ_= policyfunc(ifp,
               a_star,
               σ_star,
               discrete = False)

# + code_folding=[0]
## plot contour for policy function 

a_grid = np.linspace(0.00001,3,20)
z_grid = np.linspace(0,8,20)
aa,zz = np.meshgrid(a_grid,z_grid)

c_stars = σ_(a_grid,z_grid)
c1_stars,c2_stars = allocate(c_stars,S = 1)

fig,ax = plt.subplots(3,1,figsize=(7,8))

cp1 = ax[0].contourf(aa, zz,c_stars)
ax[0].set_title(r'$c$')
ax[0].set_xlabel('asset')
ax[0].set_ylabel('infection')


cp2 = ax[1].contourf(aa, zz,c1_stars)
ax[1].set_title(r'$c_c$')
ax[1].set_xlabel('asset')
ax[1].set_ylabel('infection')


cp3 = ax[2].contourf(aa, zz,c2_stars)
ax[2].set_title(r'$c_n$')
ax[2].set_xlabel('asset')
ax[2].set_ylabel('infection')
