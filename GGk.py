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

import math
from math import log
from math import exp
import random

def simGGk(X,mu,k,C,T=float('inf')):
    serviced = 0
    enter_times = []
    service_times = []
    interarrival_times = []
    dequeue_times = [] # sorted
    waiting_times = []
    i = 0 #total number of customers entered
    l = 0 #number of customers in the system
    t = 0 #time
    serviced = 0
    int_length = 0 # the integration of queue length w.r.t. time up to t
    idxes = np.random.choice(len(X),C)

    while t<T and i<C:
        interarrival_time = X[idxes[i]]
        service_time = np.random.exponential(1/mu)
        
        if enter_times:
            enter_times.append(enter_times[-1]+interarrival_time)
        else:
            enter_times.append(interarrival_time)
        i+=1
        interarrival_times.append(interarrival_time)
        
        # update custermers left (up to the current enter time)
        while serviced < i-1 and dequeue_times[serviced]<=enter_times[-1]: 
            tnew = min(dequeue_times[serviced],T)
            #int_length += (tnew-t)*max(l-1,0)
            t = tnew
            serviced +=1
            l-=1


        # add the new customer    
        tnew = min(enter_times[-1],T)
        t = tnew
        l+=1
  
        
        # compute the waiting time and leaving time for the new customer        
        if l>k: 
            waiting_times.append(dequeue_times[-k]-enter_times[-1])
            dequeue_times.append(dequeue_times[-k]+service_time)
        else:
            dequeue_times.append(enter_times[-1]+service_time)
            waiting_times.append(0)
        for j in range(min(k,len(dequeue_times)-1)):
            if dequeue_times[-(1+j)]<dequeue_times[-(2+j)]:
                dequeue_times[-(1+j)], dequeue_times[-(2+j)] = dequeue_times[-(2+j)], dequeue_times[-(1+j)]
        service_times.append(service_time)
#     print(np.average(waiting_times),int_length/t,t)
#     print(interarrival_times)
#     print(enter_times)
#     print(dequeue_times)
    return np.average(waiting_times[5000:])


lmbda = 1
mu = 0.4
c = 3
H = 10000

Blist = [2]
B2list = [39]
BB_list = [39]
nlist = [50]
n_rep = 1000
coverages_CB = np.zeros((len(Blist),len(nlist)))
coverages_BCB = np.zeros((len(Blist),len(B2list),len(nlist)))
coverages_BB = np.zeros((len(BB_list),len(B2list),len(nlist)))
lengthlists_CB = np.zeros((len(Blist),len(nlist),n_rep))
lengthlists_BCB = np.zeros((len(Blist),len(B2list),len(nlist),n_rep))
lengthlists_BB = np.zeros((len(BB_list),len(B2list),len(nlist),n_rep))
coverages_B = np.zeros((len(B2list),len(nlist)))
lengthlists_B = np.zeros((len(B2list),len(nlist),n_rep))
level = 0.95;
sym_level = 1-(1-level)/2;

def Psi(Xs):
#     lmbda_est = np.average(Xs)
    return simGGk(Xs,mu,c,H)
    
def generator(n):
    Z = np.random.exponential(1/lmbda,size = n)
    return Z


# estimate truth
# n_large = 100000 # number of samples used to estimate the truth
# X_large = np.random.exponential(size=(n_large,2))
# psi0 = simtrue(lmbda,mu,H)
rho = lmbda/c/mu
def C(c,rho):
    return 1/(1+(1-rho)*math.factorial(c)/pow(c*rho,c)*sum([pow(c*rho,k)/math.factorial(k) for k in range(c)]))

psi0 = C(c,rho)/(c*mu-lmbda)

# cheap bootstrap
for bidx in range(len(Blist)):
    B = Blist[bidx]
    for nidx in range(len(nlist)):
        n = nlist[nidx]
        ct = 0
        for rep in range(n_rep):
            if rep%10==0:
                print(rep)
            q = t.ppf(sym_level,B)
            Xhat = generator(n)
            psihat = Psi(Xhat)
            sumsq = 0
            for b in range(B):
                X_star_b = np.asarray(choices(Xhat,k=n))
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
                    end = time.time()
                    print(end-start)
                q = t.ppf(sym_level,B)
                Xhat = generator(n)
                psihat = Psi(Xhat)
                sumsq = 0
                for b in range(B):
                    X_star_b = np.asarray(choices(Xhat,k=n))
                    psi_star_b = Psi(X_star_b)
                    sumsq += (psi_star_b-psihat)**2
                S = math.sqrt(sumsq/B)
                if S == 0:
                    T = float('inf')
                else:
                    T = (psihat-psi0)/S

                Tstars = []
                for b2 in range(B2):
                    X_star_b2 = np.asarray(choices(Xhat,k=n))
                    psi_star_b2 = Psi(X_star_b2)
                    sumsqstar = 0
                    for b in range(B):
                        X_star_star_b2b = np.asarray(choices(X_star_b2,k=n))
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
            end = time.time()
            print(end-start)                             
# [ct/n_rep,np.average(lengths),np.std(lengths)]
# double bootstrap (additive adjustment)
for nidx in range(len(nlist)):
    n = nlist[nidx]
    for bbidx in range(len(BB_list)):
        BB = BB_list[bbidx]
        for b2idx in range(len(B2list)):
            B2 = B2list[b2idx]
            ct = 0
            for rep in range(n_rep):
                if rep%100==0:
                    print(rep)
    #             Xhat = gamma.rvs(1,size=n) 
                
                Xhat = generator(n)
                psihat = Psi(Xhat)
#                 sumsq = 0
                for b in range(BB):
                    X_star_b = np.asarray(choices(Xhat,k=n))
                    psi_star_b = Psi(X_star_b)
#                     sumsq += (psi_star_b-psihat)**2
                S = math.sqrt(sumsq/B)
                if S == 0:
                    T = float('inf')
                else:
                    T = (psihat-psi0)/S

                T_stars = [] # quantiles of psistarstars
                D_stars = [] # differences between psistars and psihat
                for b2 in range(B2):
                    X_star_b2 = np.asarray(choices(Xhat,k=n))
                    psi_star_b2 = Psi(X_star_b2)
                    sumsqstar = 0
                    Tstarstars = []
                    for b in range(BB):
                        X_star_star_b2b = np.asarray(choices(X_star_b2,k=n))
                        psi_star_star_b2b = Psi(X_star_star_b2b)
                        Tstarstars.append(abs(psi_star_star_b2b-psi_star_b2))
                    qstar = sorted(Tstarstars)[int(level*(BB+1)-1)]
                    T_stars.append(qstar)
                    D_stars.append(abs(psi_star_b2-psihat))
                Diff_stars = [D_stars[i]-T_stars[i] for i in range(len(T_stars))]
                t_hat = sorted(Diff_stars)[int(level*(B2+1)-1)]
                T = psihat - psi0
                if abs(T)<= sorted(D_stars)[int(level*(B2+1)-1)] + t_hat:
                    ct += 1
                lengthlists_BB[bbidx,b2idx,nidx,rep] = 2*(sorted(D_stars)[int(level*(B2+1)-1)] + t_hat)
                
                        #sumsqstar += (psi_star_star_b2b - psi_star_b2)**2
                    #Sstar_b2 = math.sqrt(sumsqstar/B)
            #         print(Sstar_b2)
#                     if Sstar_b2 ==0:
#                         Tstar_b2 = float('inf')
#                     else:              
#                         Tstar_b2 = (psi_star_b2-psihat)/Sstar_b2
#                     T_stars.append(abs(psi_star_star_b2b-psi_star_b2))
#                 t_hat = sorted(Bstarstars)[int(level*(B2+1)-1)]
#                 if -qhat<= T and T<=qhat:
#                     ct += 1
#                 lengthlists_BCB[bidx,b2idx,nidx,rep] = qhat*S*2
            coverages_BB[bbidx,b2idx,nidx] = ct/n_rep
end = time.time()
print(end-start)


# basic bootstrap
for nidx in range(len(nlist)):
    n = nlist[nidx]
    for b2idx in range(len(B2list)):
        B2 = B2list[b2idx]
        ct = 0
        for rep in range(n_rep):
            if rep%1000==0:
                print(rep)
#             Xhat = gamma.rvs(1,size=n) 
            Xhat = generator(n)
            psihat = Psi(Xhat)
            T_stars = []
            for b in range(B2):
                X_star_b = np.asarray(choices(Xhat,k=n))
                psi_star_b = Psi(X_star_b)
                T_stars.append(abs(psi_star_b-psihat))
            wahat = sorted(T_stars)[int(level*(B2+1)-1)]
            
            T = psihat - psi0
            if abs(T)<=wahat:
                ct += 1
            lengthlists_B[b2idx,nidx,rep] = 2*wahat
        coverages_B[b2idx,nidx] = ct/n_rep

np.savez('GGKDB04'+str(Blist[0])+'_'+str(B2list[0])+'_'+str(nlist[0]),CBprob=coverages_CB,CBlength=lengthlists_CB,BCBprob=coverages_BCB,BCBlength=lengthlists_BCB,Bprob=coverages_B,Blength=lengthlists_B,BBprob=coverages_BB,BBlength=lengthlists_BB)
