from numpy import *
from numpy.random import standard_normal, seed
import warnings
import numpy.polynomial.laguerre as a
from matplotlib.pyplot import *
from time import time
t0=time()
warnings.simplefilter('ignore', np.RankWarning)
## Simulation  Parameters
seed(150000)      # seed  for  Python  RNG
M = 50            # time  steps
I = 100000       # paths  for  valuation
reg = 7      # no of  basis  functions
AP = True          # antithetic  paths
MM = True          # moment  matching  of RN
# ## Parameters  -- American  Put  Option
r = 0.06          # short  rate
vol = 0.2           # volatility
S0 = 36.           # initial  stock  level
T = 1.0           # time -to -maturity
V0_right = 4.478 # American  Put  Option (500  steps  bin. model)
dt = T/M           # length  of time  interval
df = exp(-r*dt)   # discount  factor  per  time  interval
## Function  Definitions
def  RNG(I):
    if AP == True:
        ran=standard_normal(I//2)
        ran=concatenate ((ran,-ran))
    else:
        ran=standard_normal(I)
    if MM == True:
        ran=ran - mean(ran)
        ran=ran/std(ran)
        return ran
def  GenS(I):
    S=zeros((M+1,I),'d')           # index  level  matrix
    S[0,:]=S0                      # initial  values
    for t in range(1,M+1,1):       # index  level  paths
        ran=RNG(I)
        S[t,:]=S[t-1,:]* exp((r-vol**2/2)*dt+vol*ran*sqrt(dt))
    return S
def IV(S):
    return  maximum(40.-S,0)
## Valuation  by LSM
S=GenS(I)                     # generate  stock  price  paths
h=IV(S)                       # inner  value  matrix
V=IV(S)                       # value  matrix
for t in range(M-1,-1,-1):
    rg = polyfit(S[t,:], V[t+1,:]*df, reg)           # regression  at time t
    ##rg = a.lagfit(S[t, :], V[t + 1, :] * df, reg)
    ##C = a.lagval(S[t, :], rg, True)
    C = polyval(rg, S[t, :])
    ##C = polyval(rg,S[t,:])                            # continuation  values
    V[t,:]= where(h[t,:]>C,h[t,:],V[t+1,:]*df)   # exercise  decision
V0=sum(V[0,:])/I # LSM  estimator
print(V0)
