# -*- coding: utf-8 -*-
"""
Created: June 2022
Author: Bradley March
"""

#%% Python Preamble

# Import relevent modules
import numpy as np
import h5py
import os
cwd = os.getcwd() 

# Import user created modules
from Solvers.fR2D import fR2DSolver # 2D f(R) solver
import Packages.galaxy_relations as galf
from Packages.utils import gradient, volume_elements
from constants import M_sun, R0, G, c, rho_m

r_min = galf.r_min
r_max = galf.r_max

#%% define h5py save/load functions

def get_filename(logfR0: float, logMvir: float, N_r: int, N_th: int, cwd=cwd):
    # round to 5 decimals to prevent floating point error when saving/loading
    logMvir = np.round(logMvir, 5)
    logfR0 = np.round(logfR0, 5)
    filename = "{}_{}_{}_{}.hdf5".format(logMvir, logfR0, N_r, N_th)
    return  cwd + '/solutions/fR/' + filename

def save_solution(logfR0, logMvir, N_r, N_th, fR, cwd=cwd):
    filename = get_filename(logfR0, logMvir, N_r, N_th, cwd=cwd)
    
    file = h5py.File(filename, 'w')
    
    # set up header group
    header = file.create_group("Header")
    header.attrs['logfR0'] = logfR0
    header.attrs['logMvir'] = logMvir
    header.attrs['N_r'] = N_r
    header.attrs['N_th'] = N_th
    
    # save scalar field solution
    file.create_dataset('fR', data=fR)

    file.close()
    return

def load_solution(logfR0, logMvir, N_r, N_th, cwd=cwd):
    filename = get_filename(logfR0, logMvir, N_r, N_th, cwd=cwd)
    
    # check if solution not yet saved
    if os.path.exists(filename) is False:
        err1 = "No f(R) solution saved with parameters: "
        err2 = "logfR0={:}, logMvir={:}, N_r={:}, N_th={:}".format(
                                                    logfR0, logMvir, N_r, N_th)
        raise FileNotFoundError(err1 + err2)
    
    file = h5py.File(filename, 'r')
    fR = file['fR'][:]
    file.close()
    return fR
    

#%% solver function

def solve_field(logfR0, logMvir, N_r, N_th, cwd=cwd):    
    # define filename... 
    filename = get_filename(logfR0, logMvir, N_r, N_th, cwd=cwd)
    # ... and check if solution already exists
    if os.path.exists(filename):
        print('fR solution already exists!')
        return
    
    # derive model paramaters
    fR0 = -10**logfR0
    Mvir = 10**logMvir * M_sun

    # Setting up the grid structures
    fR_grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    # Set up the density
    rhos = galf.get_densities(Mvir, fR_grid.r, fR_grid.theta, 
                              splashback_cutoff=True, total=True)
    rho = rho_m + rhos['total']
    fR_grid.set_density(rho)
    
    # Running the solver
    fR_grid.solve(fR0=fR0, verbose=True, tol=1e-7, imin=100, imax=1000000)
    
    # calculate fR
    fR = fR_grid.usq * fR0

    save_solution(logfR0, logMvir, N_r, N_th, fR)
    
    print('logfR0={:}, logMvir={:}, solution saved'.format(logfR0, logMvir))

#%% define functions for each fR EoM term 

def delta_R(fR, fR0):
    # Calculates the delta R term in the fR EoM
    return R0 * (np.sqrt(fR0 / fR) - 1)

def delta_rho_term(drho):
    # Calculates the delta rho term in the fR EoM
    return drho * 8 * np.pi * G / c**2 

#%% Calculate fR screening radius

def calc_rs(fR, fR0, drho, grid=None, threshold=0.9, unscrthreshold=1e-3):
    
    # check if fully unscreened
    if all(fR[0, :] / fR0 > unscrthreshold):
        rs = -1
        return rs
    
    if grid is None:
        N_r, N_th = fR.shape
        grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    # determine the screened region (where lap(fR) = 0 --> dR / drho_term = 1)    
    dR = delta_R(fR, fR0)
    drho_term = delta_rho_term(drho)
    with np.errstate(divide='ignore', invalid='ignore'):
        scr_reg = np.ma.masked_invalid(dR / drho_term)
       
    # find values above a threshold 
    inds = np.argmin(scr_reg >= threshold, axis=0)

    # get screening radius
    rs = grid.r[inds, 0]
    return rs

def get_rs(logfR0, logMvir, N_r, N_th, threshold=0.9, unscrthreshold=1e-3):
    
    # load field profile
    fR = load_solution(logfR0, logMvir, N_r, N_th)

    # calculate model paramters
    fR0 = -10**logfR0
    Mvir = M_sun * 10**logMvir

    # set up grid structure
    grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
    
    # calculate density
    rhos = galf.get_densities(Mvir, grid.r, grid.theta, 
                              splashback_cutoff=True, total=True)
    
    # calculate screening radius
    rs = calc_rs(fR, fR0, rhos['total'], grid, threshold=0.9)
    return rs

#%% Binary screening condition

def critical_potential(logfR0):
    return 1.5 * 10**logfR0 * c**2

def fR0_crit_binary_screening(logMvir):  
    # From velocity dispersion # (Eqn 25 in Desmond Ferreira 2020)
    Mvir = 10**logMvir * M_sun
    dmp = galf.get_dark_matter_parameters(Mvir)
    fR0_crit = - 2 * G * Mvir / (3 * c**2 * dmp['R'])
    logfR0_crit = np.log10(abs(fR0_crit))
    return logfR0_crit

#%% Numerical approximate rs solution

def get_rs_chi(logfR0, drho, grid=None):
    # Numerically integrate chi definition to find rs

    # get critical potential
    chi = critical_potential(logfR0)
    
    if grid is None:
        N_r, N_th = drho.shape
        grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
    
    # compute necessary grid elements
    r = grid.r[:, 0]
    dr = grid.rout - grid.rin
    vol = volume_elements(grid)
    
    # average the density profile across a shell of constant theta
    average_rho = (drho * vol).sum(axis=1) / vol.sum(axis=1)
    
    # perform the numerical integration
    integrand = r * average_rho * dr[:, 0]
    cum_integral = 4 * np.pi * G * np.cumsum(integrand[::-1])[::-1]
    
    # find where integral from infinite reaches expected (at rs)
    comparison =  (cum_integral - chi) > 0
    
    # if no solution, return fully unscreened 
    if all(np.logical_not(comparison)):
        return -1
    else:
        rs_ind = np.argmin(comparison) 
        return r[rs_ind]

#%% Fifth force calculation

def get_a5(fR, fR0, grid=None, magnitude=True, scaled=False):
    
    if grid is None:
        N_r, N_th = fR.shape
        grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
        
    grad_fR = gradient(grid, fR, fR0, magnitude)
    
    if scaled:
        a5 = grad_fR / np.max(np.abs(grad_fR))
    else:
        a5 = 0.5 * c*c * grad_fR
        
    return a5

        
        
