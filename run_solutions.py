# -*- coding: utf-8 -*-
"""
Running all field solutions required to plot the figures in ScreeningPapersFigs

Created: Dec 2023
Author: Bradley March
"""

import numpy as np
from time import time 
from Packages.fR_functions import solve_field as solve_fR
from Packages.sym_functions import solve_field as solve_sym

# functions to solve arrays of parameters
def multi_fR_solver(logMvirAr, logfR0Ar, N_r=int(512), N_th=int(201)):
    for logMvir in logMvirAr:
        for logfR0 in logfR0Ar:
            solve_fR(logfR0=logfR0, logMvir=logMvir, N_r=N_r, N_th=N_th)

def multi_sym_solver(logMvirAr, logMsAr, logLcAr, N_r=int(512), N_th=int(201)):
    for logMvir in logMvirAr:
        for logMs in logMsAr:
            for logLc in logLcAr:
                solve_sym(logMs=logMs, logLc=logLc, logMvir=logMvir, 
                          N_r=N_r, N_th=N_th)

# Grid resolution (figs 1-4)
N_r = int(512)
N_th = int(201)

#%% Figure 1 solutions
starttime = time()

logMvir = 11.5 
logfR0 = -6.4
logMs = -4.5
logLc = -1

solve_fR(logfR0=logfR0, logMvir=logMvir, N_r=N_r, N_th=N_th)
solve_sym(logMs=logMs, logLc=logLc, logMvir=logMvir, N_r=N_r, N_th=N_th)

endtime = time()
print("Figure 1 solutions took {:.2f}".format(endtime-starttime))

#%% Figure 2 solutions
starttime = time()

# f(R)
dMvir, dfR0 = 0.2, 0.2
logfR0Ar = np.arange(-8, -5+dfR0/2, dfR0)
logMvirAr = np.arange(10, 13.5+dMvir/2, dMvir)
multi_fR_solver(logMvirAr, logfR0Ar)

# sym
dMvir, dMs, dLc = 0.5, 0.5, 0.5
logMvirAr = np.arange(10, 13.5+dMvir/2, dMvir)
logMsAr = np.arange(-6.5, -3+dMs/2, dMs)
logLcAr = np.arange(-3, 3+dLc/2, dLc)
multi_sym_solver(logMvirAr, logMsAr, logLcAr)

endtime = time()
print("Figure 2 solutions took {:.2f}".format(endtime-starttime))

#%% Figure 3 solutions
starttime = time()

# f(R)
logMvirAr = np.array([11.5])
logfR0Ar = np.array([-6.3, -6.4, -6.5, -6.6, -6.8, -7.0, -7.2])
multi_fR_solver(logMvirAr, logfR0Ar)

# sym (Lc fixed)
logMsAr = np.array([-4.2, -4.6, -4.8, -4.9, -5.1, -5.5, -5.8, -6.0])
logLcAr = np.array([-1.])
multi_sym_solver(logMvirAr, logMsAr, logLcAr)

# sym (Ms fixed)
logLcAr = np.array([-1.5, -0.9, -0.7, -0.6, -0.4, 0.0, 0.3, 0.5])
logMsAr = np.array([-4.5])
multi_sym_solver(logMvirAr, logMsAr, logLcAr)

endtime = time()
print("Figure 3 solutions took {:.2f}".format(endtime-starttime))

#%% Figure 4 solutions
starttime = time()

# f(R)
dMvir, dfR0 = 0.1, 0.1
logfR0Ar = np.arange(-8, -5+dfR0/2, dfR0)
logMvirAr = np.arange(10, 13.5+dMvir/2, dMvir)
multi_fR_solver(logMvirAr, logfR0Ar)

# sym (Lc fixed)
dMs = 0.1
logMsAr = np.arange(-6.5, -3+dMs/2, dMs)
logLcAr = np.array([-1.])
multi_sym_solver(logMvirAr, logMsAr, logLcAr)

# sym (Ms fixed)
dLc = 0.1
logLcAr = np.arange(-3, 2+dLc/2, dLc) # Larger than 2 => no SSB
logMsAr = np.array([-4.5])
multi_sym_solver(logMvirAr, logMsAr, logLcAr)

endtime = time()
print("Figure 4 solutions took {:.2f}".format(endtime-starttime))
    

#%% Figure 5 solutions
starttime = time()

# MW mass
logMvirAr = np.array([np.log10(1.5e12)]) 

# f(R)
dfR0 = 0.05
logfR0Ar = np.arange(-8, -5+dfR0/2, dfR0)
multi_fR_solver(logMvirAr, logfR0Ar, N_th=int(101))

# f(R) inset
dfR0 = 0.01
logfR0Ar = np.arange(-6.2, -5.8, dfR0)
multi_fR_solver(logMvirAr, logfR0Ar, N_th=int(101))

# sym
dMs, dLc = 0.1, 0.1
logMsAr = np.arange(-6.5, -3+0.01, dMs)
logLcAr = np.arange(-3, 3+0.01, dLc)
multi_sym_solver(logMvirAr, logMsAr, logLcAr, N_th=int(101))

endtime = time()
print("Figure 5 solutions took {:.2f}".format(endtime-starttime))
    






