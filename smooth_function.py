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

Blist = [1,2,3,4,5,10,20,30]
B2list = [39,79]
nlist = [25,50,100,200,400]
n_rep = 10000
coverages_CB = np.zeros((len(Blist),len(nlist)))
coverages_BCB = np.zeros((len(Blist),len(B2list),len(nlist)))
coverages_BA = np.zeros((len(Blist),len(nlist)))
coverages_BBA = np.zeros((len(Blist),len(B2list),len(nlist)))
lengthlists_CB = np.zeros((len(Blist),len(nlist),n_rep))
lengthlists_BCB = np.zeros((len(Blist),len(B2list),len(nlist),n_rep))
lengthlists_BA = np.zeros((len(Blist),len(nlist),n_rep))
lengthlists_BBA = np.zeros((len(Blist),len(B2list),len(nlist),n_rep))
coverages_B = np.zeros((len(Blist),len(B2list),len(nlist)))
lengthlists_B = np.zeros((len(Blist),len(B2list),len(nlist),n_rep))
level = 0.95
sym_level = 1-(1-level)/2

def Psi(Xs):
    m = np.sum(Xs,axis=0)/len(Xs)
    return math.sin(m[0]+m[1]**2/5)

# estimate truth
n_large = 100000 # number of samples used to estimate the truth
X_large = np.random.exponential(size=(n_large,2))
psi0 = Psi(X_large)


start = time.time()
# cheap bootstrap
for bidx in range(len(Blist)):
    B = Blist[bidx]
    for nidx in range(len(nlist)):
        n = nlist[nidx]
        ct = 0
        for rep in range(n_rep):
            if rep%1000==0:
                print([B,n,rep])
            q = t.ppf(sym_level,B)
            Xhat = np.random.exponential(size=(n,2))
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

for bidx in range(len(Blist)):
    B = Blist[bidx]
    for b2idx in range(len(B2list)):
        B2 = B2list[b2idx]        
        for nidx in range(len(nlist)):
            n = nlist[nidx]
            

            ct = 0
            for rep in range(n_rep):
                if rep%400==0:
                    print([B,n,rep])
                q = t.ppf(sym_level,B)
                Xhat = np.random.exponential(size=(n,2))
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


# standard bootstrap with B*B2 resamples (to match the computational effort)
for bidx in range(len(Blist)):
    B = Blist[bidx]
    for b2idx in range(len(B2list)):
        B2 = B2list[b2idx]        
        for nidx in range(len(nlist)):
            n = nlist[nidx]
            ct = 0
            for rep in range(n_rep):
                if rep%1000==0:
                    print([B,n,rep])
    #             Xhat = gamma.rvs(1,size=n) 
                Xhat = np.random.exponential(size=(n,2))
                psihat = Psi(Xhat)
                T_stars = []
                for b in range(B2*B):
                    X_star_b = choices(Xhat,k=n)
                    psi_star_b = Psi(X_star_b)
                    T_stars.append(psi_star_b-psihat)
                wahat = sorted(T_stars)[int(sym_level*(B2*B+1)-1)]
                wahat2 = sorted(T_stars)[int((1-sym_level)*(B2*B+1)-1)]
                T = psihat - psi0
                if wahat2<=T and T<=wahat:
                    ct += 1
                lengthlists_B[bidx,b2idx,nidx,rep] = wahat - wahat2
            coverages_B[bidx,b2idx,nidx] = ct/n_rep
            
# batching (not corrected)
for bidx in range(len(Blist)):
    B = Blist[bidx]
    for nidx in range(len(nlist)):
        n = nlist[nidx]
        if B==1 or n < B*4: #num of batches too few, or the sample size per batch too small
            continue
        ct = 0
        for rep in range(n_rep):
            if rep%1000==0:
                print([B,n,rep])
            q = t.ppf(sym_level,B-1)
            bsize = n//B
            Xhat = np.random.exponential(size=(n,2))
            psihat = Psi(Xhat)
            sumsq = 0
            for b in range(B):
                X_star_b = Xhat[(b*bsize):((b+1)*bsize)]
                psi_star_b = Psi(X_star_b)
                sumsq += (psi_star_b-psihat)**2
            S = math.sqrt(sumsq/(B-1))
            if S == 0:
                T = float('inf')
            else:
                T = (psihat-psi0)/S
            if -q<= T and T<=q:
                ct += 1
            lengthlists_BA[bidx,nidx,rep] = q*S*2
        coverages_BA[bidx,nidx] = ct/n_rep
        
        
# corrected batching

for bidx in range(len(Blist)):
    B = Blist[bidx]
    for b2idx in range(len(B2list)):
        B2 = B2list[b2idx]        
        for nidx in range(len(nlist)):
            n = nlist[nidx]
            
            if B==1 or n < B*4: #num of batches too few, or the sample size per batch too small
                continue
            ct = 0
            
            bsize = n//B
            for rep in range(n_rep):
                if rep%400==0:
                    print([B,n,rep])
                q = t.ppf(sym_level,B-1)
                Xhat = np.random.exponential(size=(n,2))
                psihat = Psi(Xhat)
                sumsq = 0
                for b in range(B):
                    X_star_b = choices(Xhat,k=n)
                    psi_star_b = Psi(X_star_b)
                    sumsq += (psi_star_b-psihat)**2
                S = math.sqrt(sumsq/(B-1))
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
                        X_star_star_b2b = X_star_b2[(b*bsize):((b+1)*bsize)]
                        psi_star_star_b2b = Psi(X_star_star_b2b)
                        sumsqstar += (psi_star_star_b2b - psi_star_b2)**2
                    Sstar_b2 = math.sqrt(sumsqstar/(B-1))
            #         print(Sstar_b2)
                    if Sstar_b2 ==0:
                        Tstar_b2 = float('inf')
                    else:              
                        Tstar_b2 = (psi_star_b2-psihat)/Sstar_b2
                    Tstars.append(abs(Tstar_b2))
                qhat = sorted(Tstars)[int(level*(B2+1)-1)]
                if -qhat<= T and T<=qhat:
                    ct += 1
                lengthlists_BBA[bidx,b2idx,nidx,rep] = qhat*S*2
            coverages_BBA[bidx,b2idx,nidx] = ct/n_rep

end = time.time()
print(end-start)        


np.savez('results_exp',CBprob=coverages_CB,CBlength=lengthlists_CB,BCBprob=coverages_BCB,BCBlength=lengthlists_BCB,
        Bprob = coverages_B, Blength=lengthlists_B,BAprob=coverages_BA,BAlength=lengthlists_BA,BBAprob = coverages_BBA, BBAlength = lengthlists_BBA)