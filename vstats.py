import numpy as np
from numpy.linalg import inv
import math
from scipy.stats import t
from scipy.stats import norm
import itertools
import time
from scipy.stats import chi2
from scipy.stats import gamma
from random import choices
import matplotlib.pyplot as plt
import time

Blist = [3,4]
B2list = [39,79]
nlist = [10,20,30]
n_rep = 10000
coverages_CB = np.zeros((len(Blist),len(nlist)))
coverages_BCB = np.zeros((len(Blist),len(B2list),len(nlist)))
lengthlists_CB = np.zeros((len(Blist),len(nlist),n_rep))
lengthlists_BCB = np.zeros((len(Blist),len(B2list),len(nlist),n_rep))
level = 0.95;
sym_level = 1-(1-level)/2;

def h(x,y):
    return min(12,(x-y)**2+x+y)
def Psi(Xs):
    return sum([sum ([h(x,y) for x in Xs]) for y in Xs])/(len(Xs)**2)

# estimate truth
n_large = 10000 # number of samples used to estimate the truth
X_large = gamma.rvs(1,size=n_large)
psi0 = Psi(X_large)

# cheap bootstrap
for bidx in range(len(Blist)):
    B = Blist[bidx]
    for nidx in range(len(nlist)):
        n = nlist[nidx]
        ct = 0
        for rep in range(n_rep):
            if rep%1000==0:
                print(rep)
            q = t.ppf(sym_level,B)
            Xhat = gamma.rvs(1,size=n) 
            psihat = Psi(Xhat)
            sumsq = 0
            for b in range(B):
                X_star_b = choices(Xhat,k=n)
                psi_star_b = Psi(X_star_b)
                sumsq += (psi_star_b-psihat)**2
            S = math.sqrt(sumsq/B)
            if S == 0:
                T = float('inf')
            else:
                T = (psihat-psi0)/S
            if -q<= T and T<=q:
                ct += 1
            lengthlists_CB[bidx,nidx,rep] = q*S*2
        coverages_CB[bidx,nidx] = ct/n_rep
        
# bootstrap corrected cheap bootstrap
start = time.time()
for bidx in range(len(Blist)):
    B = Blist[bidx]
    for b2idx in range(len(B2list)):
        B2 = B2list[b2idx]        
        for nidx in range(len(nlist)):
            n = nlist[nidx]
            ct = 0
            for rep in range(n_rep):
                if rep%400==0:
                    print(rep)
                q = t.ppf(sym_level,B)
                Xhat = gamma.rvs(1,size=n) 
                psihat = Psi(Xhat)
                sumsq = 0
                for b in range(B):
                    X_star_b = choices(Xhat,k=n)
                    psi_star_b = Psi(X_star_b)
                    sumsq += (psi_star_b-psihat)**2
                S = math.sqrt(sumsq/B)
                if S == 0:
                    T = float('inf')
                else:
                    T = (psihat-psi0)/S

                Tstars = []
                for b2 in range(B2):
                    X_star_b2 = choices(Xhat,k=n)
                    psi_star_b2 = Psi(X_star_b2)
                    sumsqstar = 0
                    for b in range(B):
                        X_star_star_b2b = choices(X_star_b2,k=n)
                        psi_star_star_b2b = Psi(X_star_star_b2b)
                        sumsqstar += (psi_star_star_b2b - psi_star_b2)**2
                    Sstar_b2 = math.sqrt(sumsqstar/B)
            #         print(Sstar_b2)
                    if Sstar_b2 ==0:
                        Tstar_b2 = float('inf')
                    else:              
                        Tstar_b2 = (psi_star_b2-psihat)/Sstar_b2
                    Tstars.append(abs(Tstar_b2))
                qhat = sorted(Tstars)[int(level*(B2+1)-1)]
                if -qhat<= T and T<=qhat:
                    ct += 1
                lengthlists_BCB[bidx,b2idx,nidx,rep] = qhat*S*2
            coverages_BCB[bidx,b2idx,nidx] = ct/n_rep
# [ct/n_rep,np.average(lengths),np.std(lengths)]
end = time.time()
print(end-start)
np.savez('results',CBprob=coverages_CB,CBlength=lengthlists_CB,BCBprob=coverages_BCB,BCBlength=lengthlists_BCB)