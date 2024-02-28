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
from Solvers.symm2D import Symm2DSolver
import Packages.galaxy_relations as galf
from Packages.utils import gradient, laplacian
from constants import kpc, M_sun, c, G, rho_m

r_min = galf.r_min
r_max = galf.r_max

#%% define h5py save/load functions

def get_filename(logMs: (float, int), logLc: (float, int), logMvir: (float, int), 
                 N_r: int, N_th: int, cwd=cwd) -> str:
    # round to 5 decimals to prevent floating point error when saving/loading
    logMs = np.round(float(logMs), 5)
    logLc = np.round(float(logLc), 5)
    logMvir = np.round(float(logMvir), 5)
    filename = "{}_{}_{}_{}_{}..hdf5".format(logMvir, logMs, logLc, N_r, N_th)
    return  cwd + '/solutions/sym/' + filename

def save_solution(logMs, logLc, logMvir, N_r, N_th, u, u_inf, cwd=cwd):
    filename = get_filename(logMs, logLc, logMvir, N_r, N_th, cwd=cwd)
    
    file = h5py.File(filename, 'w')
    
    # set up header group
    header = file.create_group("Header")
    header.attrs['logMs'] = logMs
    header.attrs['logLc'] = logLc
    header.attrs['logMvir'] = logMvir
    header.attrs['u_inf'] = u_inf
    header.attrs['N_r'] = N_r
    header.attrs['N_th'] = N_th
    
    # save scalar field solution
    file.create_dataset('u', data=u)

    file.close()
    return

def load_solution(logMs, logLc, logMvir, N_r, N_th, cwd=cwd):
    filename = get_filename(logMs, logLc, logMvir, N_r, N_th, cwd=cwd)
    
    # check if solution already exists
    if os.path.exists(filename) is False:
        err1 = "No symmetron solution saved with parameters: "
        err2 = "logMs={:}, loglams={:}, logM_vir={:}, N_r={:}, N_th={:}".format(
                        logMs, logLc, logMvir, N_r, N_th)
        raise FileNotFoundError(err1 + err2)
    
    # open file and extract fR solution
    file = h5py.File(filename, 'r')
    header = file["Header"]
    u = file['u'][:]
    u_inf = u_inf = header.attrs['u_inf']
    file.close()
    
    return u, u_inf

#%% Solver function

def solve_field(logMs, logLc, logMvir, N_r, N_th, cwd=cwd):    
    # define filename... 
    filename = get_filename(logMs, logLc, logMvir, N_r, N_th, cwd=cwd)
    # ... and check if solution already exists
    if os.path.exists(filename):
        print('sym solution already exists!')
        return
    
    # derive model paramaters
    Lc = 10**(logLc) * kpc
    Ms = 10**(logMs) # solvers scales by Planck mass
    Mvir = 10**logMvir * M_sun
    
    # Setting up the grid structures
    sym_grid = Symm2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    # Set up the density
    rhos = galf.get_densities(Mvir, sym_grid.r, sym_grid.theta, 
                         splashback_cutoff=True, total=True)
    rho = rhos['total'] + rho_m
    sym_grid.set_density(rho)
    
    
    # Running the solver
    sym_grid.solve(Ms, Lc, set_zero_region=True, verbose=True, 
               tol=1e-14, imin=100, imax=1000000)
    
    save_solution(logMs, logLc, logMvir, N_r, N_th, sym_grid.u, sym_grid.u_inf)
    
    print('logMs={:}, logLc={:}, logMvir={:}, solution saved'.format(
            logMs, logLc, logMvir))
    return 

#%% Calculate sym screening radius

def calc_rs(u, u_inf, grid=None, 
            threshold=1e-1, unscrthreshold=1, unscrlapthreshold=1e-1):
    # check if SSB has happened yet
    if u_inf == 0:
        rs = np.inf
        return rs
        
    # check if field is fully unscreened
    if any(u[0, :] / u_inf > unscrthreshold):
        rs = -1
        return rs
    
    if grid is None:
        N_r, N_th = u.shape()
        grid = Symm2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
    
    # check if field evades our previous fully unscreened definiton
    lap_u = laplacian(grid, u, u_inf) 
    if any(lap_u[0, :]/lap_u.max() > unscrlapthreshold):
        rs = -2 
        return rs

    # find field values below a threshold 
    inds = np.argmin(u / u_inf <= threshold, axis=0)
    # get screening radius
    rs = grid.r[inds, 0]        
    return rs

def get_rs(logMs, logLc, logMvir, N_r, N_th,
           threshold=1e-1, unscrthreshold=1, unscrlapthreshold=1e-1):
    
    # load sym solution 
    u, u_inf = load_solution(logMs, logLc, logMvir, N_r, N_th)

    # set up grid structure
    grid = Symm2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
    
    # calculate rs
    rs = calc_rs(u, u_inf, grid, threshold=threshold, 
                 unscrthreshold=unscrthreshold, 
                 unscrlapthreshold=unscrlapthreshold)
    
    return rs

#%% Binary screening condition

def critical_potential(logMs):
    crit_pot = 0.5 * (10**logMs)**2 * c*c
    return crit_pot

def logMs_crit_binary_screening(logMvir):
    Mvir = 10**logMvir * M_sun
    dmp = galf.get_dark_matter_parameters(Mvir)
    Ms_crit = np.sqrt(2 * G * Mvir / dmp['R']) / c
    logMs_crit = np.log10(Ms_crit)
    return logMs_crit

#%% Numerical approximate rs solution

def get_rho_SSB(logMs, logLc):
    Ms = 10**logMs
    Lc = kpc * 10**logLc
    rho_SSB = (c*c / (16 * np.pi * G)) * (Ms / Lc)**2
    return rho_SSB

def get_rho_SSB_rs(logMs, logLc, drho, grid=None):
   
    # check if SSB has occured yet
    rho_SSB = get_rho_SSB(logMs, logLc)
    if rho_SSB < rho_m:
        return np.inf

    # find where rho < rho_SSB       
    rho = drho + rho_m
    scr_bool = rho < rho_SSB
    
    # determine if the galaxy is fully screened
    if scr_bool.all():
        return -1
    else:  
        if grid is None:
            N_r, N_th = drho.shape
            grid = Symm2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
        rs_inds = np.argmax(scr_bool, axis=0)
        rs = grid.r[:, 0][rs_inds]
        
    return rs

#%% Fifth force calculation

def get_a5(u, u_inf, grid=None, magnitude=True, scaled=False):
    # This calculation should have a prefactor, but since it has a 1/lambda in 
    # we instead only give here an unscaled version (to scale by max a5)
    if grid is None:
        N_r, N_th = u.shape
        grid = Symm2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

    grad_u = gradient(grid, u, u_inf, magnitude)
    a5 = - u * grad_u
    if scaled:
        a5 = a5 / np.max(np.abs(a5))
    return a5
