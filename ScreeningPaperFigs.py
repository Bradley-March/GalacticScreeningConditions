# -*- coding: utf-8 -*-
"""
Code to create the figures of screening conditions paper (2310.19955)

Created: June 2022
Author: Bradley March
"""
#%% Python Preamble

# import relevent modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects
import time

# import user created modules
from Solvers.fR2D import fR2DSolver # needed to create grid structure
import Packages.fR_functions as fRf
import Packages.sym_functions as symf
import Packages.galaxy_relations as galf
from Packages.utils import mass_enclosed
from constants import kpc, M_sun, rho_m

# plotting parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['image.cmap'] = 'inferno'
cmap = plt.colormaps.get_cmap(plt.rcParams['image.cmap'])
size = 6
dpi = 300
savefigure = False

# grid structure inputs:
N_r = int(512)
N_th = int(201)
r_min = galf.r_min
r_max = galf.r_max
# set up grid structure to use for figures 1-4
grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

#%% Plotting functions

def get_full_polars(th, r, fg, extend_theta=True, rmaxplotted=None):
    """Takes polar coordinates (th, r) and value to be plotted (fg). 
    extend_theta: Extends over the theta coordinate (assuming periodicity at 
    boundary) to plot the full extent of the semi-circle.
    rmaxplotted: Cuts off values at the max radius."""
    
    # convert coordinates to vectors (if grid-like)
    if th.ndim == 2:
        th = th[0, :]
    if r.ndim == 2:
        r = r[:, 0]
        
    # add extra theta coordinate to plot over full semi-circle
    if extend_theta:
        th = np.hstack((th[-1] + np.pi, th, th[0] - np.pi))
        fg = np.hstack((fg[:, -1][:, None], fg, fg[:, 0][:, None]))
        
    # cut off the radial extent at rmaxplotted
    if rmaxplotted is not None:
        mask = r < rmaxplotted
        r = r[mask]
        fg = fg[mask, :]
        
    return th, r, fg


def bool_boundary_divider(data_bool, x, y, ax, linecolor='r', linewidth=1):
    """Takes 2D boolean array and an axis with the same shape and plots a 
    all dividing lines between 0 and 1 values."""
    
    # assumes constant spacing in x, y
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # find indices where there is a jump from 0-1 or 1-0
    horizontal_inds = np.where(abs(np.diff(data_bool, axis=0)) == 1)
    vertical_inds = np.where(abs(np.diff(data_bool, axis=1)) == 1)
    
    # convert indices to line coordinates
    line_coords = []  
    for y_ind, x_ind in zip(horizontal_inds[0], horizontal_inds[1]):
        line_coords.append([(x[x_ind]-dx/2, x[x_ind]+dx/2), 
                            (y[y_ind]+dy/2, y[y_ind]+dy/2)])
    for y_ind, x_ind in zip(vertical_inds[0], vertical_inds[1]):
        line_coords.append([(x[x_ind]+dx/2, x[x_ind]+dx/2), 
                            (y[y_ind]-dy/2, y[y_ind]+dy/2)])
    
    # plot dividing lines
    for line in line_coords:
        ax.plot(np.array(line[0]), np.array(line[1]), 
                color=linecolor, linewidth=linewidth)

#%% Fig 1

def plot_figure_1(logMvir=11.5, logfR0=-6.4, logMs=-4.5, logLc=-1, grid=grid,
                  N_stellar_scale_length_plotted=5, N_grid_lines_plotted=5,
                  savefigure=False):
    """Plots figure 1 -- 2x2 figure with panels representing:
    log density                        ; grid coordinates
    fR field & curvature-density param ; sym field & restricted sym field"""
    starttime = time.time()

    # derive model parameters
    Mvir = 10**logMvir * M_sun
    fR0 = - 10**logfR0
    
    # get non-background density
    drho = galf.get_densities(Mvir, grid.r, grid.theta, 
                              splashback_cutoff=True, total=True)['total']
    
    # load fR/sym solution 
    fR = fRf.load_solution(logfR0, logMvir, N_r, N_th)
    u, u_inf = symf.load_solution(logMs, logLc, logMvir, N_r, N_th)
    
    # calculate fR EoM term
    eom = fRf.get_curvature_density_ratio(fR, fR0, drho)
    
    # set up figure
    asp = 5/3
    fig = plt.figure(figsize=(size, size / asp))
    
    # panel sizes
    dX = 0.25
    dY = 0.4
    # panel positions
    X1L, X2L = 0.0755, 0.559
    semicircle_width = 0.12235
    X1R, X2R = X1L + semicircle_width, X2L + semicircle_width
    Y2 = 0.02
    Y1 = Y2 + dY + 0.1 
    # create axes
    ax12 = fig.add_axes([X1L + 0.063, Y1, dX, dY], projection='polar')
    ax3 = fig.add_axes([X2L, Y1, dX, dY], projection='polar')
    ax4 = fig.add_axes([X2R, Y1, dX, dY], projection='polar')
    ax5 = fig.add_axes([X1L, Y2, dX, dY], projection='polar')
    ax6 = fig.add_axes([X1R, Y2, dX, dY], projection='polar')
    ax7 = fig.add_axes([X2L, Y2, dX, dY], projection='polar')
    ax8 = fig.add_axes([X2R, Y2, dX, dY], projection='polar')
    axes = [ax12, ax3, ax4, ax5, ax6, ax7, ax8]
    
    # colourbar positions and axes
    fullcircle_width = 0.28
    cX1L, cX2L = 0.106, 0.59
    cX1R, cX2R = cX1L + fullcircle_width, cX2L + fullcircle_width
    cdX, cdY = 0.03, 0.39
    cY1, cY2 = Y1 + (dY-cdY)/2, Y2 + (dY-cdY)/2
    cax1 = fig.add_axes([cX1L, cY1, cdX, cdY])
    cax5 = fig.add_axes([cX1L, cY2, cdX, cdY])
    cax6 = fig.add_axes([cX1R, cY2, cdX, cdY])
    cax7 = fig.add_axes([cX2L, cY2, cdX, cdY])
    cax8 = fig.add_axes([cX2R, cY2, cdX, cdY])
    caxes = [cax1, cax5, cax6, cax7, cax8]
    ims = [] # to append images for colourbar input
    
    # title positions and axes
    tX1, tX2 = 0, 0.5
    tY1, tY2 = 0.95, 0.45
    tdX, tdY = 0.5, 0.0
    tax1 = fig.add_axes([tX1, tY1, tdX, tdY])
    tax2 = fig.add_axes([tX2, tY1, tdX, tdY])
    tax3 = fig.add_axes([tX1, tY2, tdX, tdY])
    tax4 = fig.add_axes([tX2, tY2, tdX, tdY])
    taxes = [tax1, tax2, tax3, tax4]
    titles = ['Density Profile', 'Coordinate Grid',
              r'$f(R)$ Solution', 'Symmetron Solution']
    # remove axis from boxes and add text
    for ind, ax in enumerate(taxes):
        ax.axis('off')
        ax.text(0.5, 0.5, s=titles[ind], ha='center')
    
    # set general properties for panel axes
    for LR_ind, ax in enumerate(axes):
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_theta_direction((-1)**(LR_ind+1))
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        
    # set density axis to be full circle
    ax12.set_thetamax(360)
    
    # define max radial extent
    stellar_scale_length = galf.get_stellar_disc_parameters(Mvir)['R']
    rmaxplotted = N_stellar_scale_length_plotted * stellar_scale_length
    
    r = grid.r[:, 0]
    th = grid.theta[0, :]
    
    # plot 1: density 
    # whole circle: density
    # calculate log density (masking invalid values from zero density)
    logrho = np.ma.log10(drho/rho_m)
    # cutoff r and logrho at rmaxplotted
    the, re, logrhoe = get_full_polars(th, r, logrho, extend_theta=False, 
                                       rmaxplotted=rmaxplotted)
    # extend th and rho over full circle
    the = np.hstack((the[-1] + 2*np.pi, the + np.pi, the))
    logrhoe = np.hstack((logrhoe[:, -1][:, None], logrhoe[:, ::-1], logrhoe))
    # plot density and save im
    im1 = ax12.pcolormesh(the, re/kpc, logrhoe, shading='auto')
    ims.append(im1)
    
    # plot 2: grid coordinates
    # left: radial coordinates 
    ax3.grid(True, axis='x')
    ax3.set_xticks(th[::N_grid_lines_plotted])
    ax3.tick_params(grid_linewidth=.5, grid_color='black')
    # add disc plane line
    ax3.axvline(np.pi/2, linestyle='dashed', linewidth=2, color=cmap(0.3))
    temp_ax = fig.add_axes([0.47, 0.52, 0.2, 0.4])
    temp_ax.axis('off')
    temp_ax.annotate(r'Disc Plane', xy=(0, 0), xytext=(0, 0.45), 
                     color=cmap(0.3))
    # right: angular coordinates
    ax4.grid(True, axis='y')
    ax4.set_yticks(r[::N_grid_lines_plotted]/kpc)
    ax4.tick_params(grid_linewidth=.5, grid_color='black')  
    # add arrows to show 5R_disc 
    temp_ax = fig.add_axes([0.87, 0.52, 0.2, 0.4])
    temp_ax.axis('off')
    temp_ax.annotate('', xy=(0.1, 1), xytext=(0.1, 0.45),
                     arrowprops=dict(arrowstyle='<->', linewidth=2, 
                                     color=cmap(0.7)))
    temp_ax.annotate(r'5$R_\mathrm{disc}$', xy=(0, 0), xytext=(0.13, 0.6), 
                     rotation=270)
    
    # plot 3: fR solution
    # left: fR field profile
    # get extended and cutoff polar coords
    the, re, usqe = get_full_polars(th, r, fR/fR0, extend_theta=True,
                                    rmaxplotted=rmaxplotted)
    im5 = ax5.pcolormesh(the, re/kpc, usqe, shading='auto')
    ims.append(im5)
    # right: curvature-density screening condition
    _, _, eome = get_full_polars(th, r, eom, extend_theta=True, 
                                 rmaxplotted=rmaxplotted)
    im6 = ax6.pcolormesh(the, re/kpc, eome, shading='auto', vmin=0, vmax=1)
    ims.append(im6)
    
    # plot 4: sym solution
    # left: sym field profile
    the, re, ue = get_full_polars(th, r, u/u_inf, extend_theta=True,
                                rmaxplotted=rmaxplotted)
    im7 = ax7.pcolormesh(the, re/kpc, ue, shading='auto', vmin=0, vmax=1)
    ims.append(im7)
    # right: sym threshold screening condition
    # add colourmap cutoff of 0.1 to show screening condition
    im8 = ax8.pcolormesh(the, re/kpc, ue, shading='auto', vmin=0, vmax=0.1)
    ims.append(im8)
    
    # plot colourbars
    clabels = [r"$\log_{10}(\rho\ /\ \bar{\rho})$", 
               r"$f_R\ /\ f_{R0}$",  
               r'$\delta R\ /\ 8\pi G\delta\rho/c^2$', 
               r"$\varphi\ /\ \varphi_\infty$", 
               r"$\varphi\ /\ \varphi_\infty$"]
    fmt = "%4.1g" 
    ticklocations = ['left', 'right']
    rotations = [90, 270]
    labelpads = [1, 12]
    extend = 'neither'
    for ind, cax in enumerate(caxes):
        im = ims[ind]
        label = clabels[ind]
        if cax is not cax1:
            ind += 1
        if cax is cax8:
            extend = 'max'
        cbar = plt.colorbar(im, cax=cax, format=fmt, 
                            ticklocation=ticklocations[ind%2], extend=extend)
        cbar.ax.set_ylabel(label, rotation=rotations[ind%2], 
                           labelpad=labelpads[ind%2])
        
    plt.show()
    
    if savefigure is True:
        plt.savefig('Fig_1._rho,_fR,_sym_and_scr_conds_new.png', dpi=dpi)
        print('Fig 1. saved!')
    
    fintime = time.time()
    print('Fig. 1 took {:.2f}s'.format(fintime-starttime))

#%% Fig 2 (with a5 plots, 2x2 grid)

def plot_figure_2(dfR0=0.2, fR_dMvir=0.2, dMs=0.5, dLc=0.5, sym_dMvir=0.5, 
                  grid=grid, fR_threshold=0.9, fR_unscrthreshold=1e-3,
                  sym_threshold=0.1, sym_unscrfieldthreshold=1e-3,
                  sym_unscrlapthreshold=1e-1, savefigure=False):
    """Plots figure 2 -- 2x2 figure with panels representing:
    f(R) screening condition        ; f(R) fifth force
    symmetron screening condition   ; symmetron fifth force"""
    starttime = time.time()
    
    # f(R) parameters
    logfR0_range = np.arange(-8, -5+dfR0/2, dfR0)
    fR_logMvir_range = np.arange(10, 13.5+fR_dMvir/2, fR_dMvir)
    # Symmeton parameters
    logMs_range = np.arange(-6.5, -3+dMs/2, dMs)
    logLc_range = np.arange(-3, 3+dLc/2, dLc)
    sym_logMvir_range = np.arange(10, 13.5+sym_dMvir/2, sym_dMvir)
    
    # set up grid structure
    th_ind = grid.disc_idx
    r = grid.r[:, th_ind]
    
    # set up empty fR/sym arrays
    fR_rs_all = np.zeros([fR_logMvir_range.size, logfR0_range.size])
    fR_EoM_all = np.zeros([fR_logMvir_range.size, logfR0_range.size, N_r])
    fR_a5_all = np.zeros_like(fR_EoM_all)
    sym_rs_all = np.zeros([sym_logMvir_range.size, logMs_range.size, 
                           logLc_range.size])
    sym_field_all = np.zeros([sym_logMvir_range.size, logMs_range.size, 
                              logLc_range.size, N_r])
    sym_a5_all = np.zeros_like(sym_field_all)
    
    # Collect fR solutions
    for i, logMvir in enumerate(fR_logMvir_range):
        Mvir = M_sun * 10**logMvir
        drho = galf.get_densities(Mvir, grid.r, grid.theta, total=True,
                                  splashback_cutoff=True)['total'][:, th_ind]
        for j, logfR0 in enumerate(logfR0_range):
            fR0 = -10**logfR0
            
            # get rs
            rs = fRf.get_rs(logfR0, logMvir, N_r, N_th, threshold=fR_threshold,
                            unscrthreshold=fR_unscrthreshold)
            # check if fully unscreened
            if isinstance(rs, int):
                fR_rs_all[i, j] = rs
                continue
            fR_rs_all[i, j] = rs[th_ind]
            
            # load fR field
            fR = fRf.load_solution(logfR0, logMvir, N_r, N_th)
            
            # get fR a5
            a5 = fRf.get_a5(fR, fR0, magnitude=True)[:, th_ind]
            fR_a5_all[i, j] = np.abs(a5) / np.abs(a5).max()
            
            # get fR curvature-density ratio
            fR_EoM_all[i, j] = fRf.get_curvature_density_ratio(fR[:, th_ind], 
                                                               fR0, drho)
            
    # Collect symmetron solutions
    for i, logMvir in enumerate(sym_logMvir_range):
        for j, logMs in enumerate(logMs_range):
            for k, logLc in enumerate(logLc_range):
                
                # get rs
                rs = symf.get_rs(logMs, logLc, logMvir, N_r, N_th,
                                 threshold=sym_threshold, 
                                 unscrthreshold=sym_unscrfieldthreshold, 
                                 unscrlapthreshold=sym_unscrlapthreshold)
                # check if fully screened/unscreened
                if isinstance(rs, int) or rs is np.inf:
                    sym_rs_all[i, j, k] = rs 
                    continue
                sym_rs_all[i, j, k] = rs[th_ind]
                
                # load sym field
                u, u_inf = symf.load_solution(logMs, logLc, logMvir, N_r, N_th)
                sym_field_all[i, j, k] = u[:, th_ind] / u_inf
                
                # get sym a5
                a5 = symf.get_a5(u, u_inf, magnitude=True)[:, th_ind]
                sym_a5_all[i, j, k] = np.abs(a5) / np.abs(a5).max()     
    
    # set up figure
    asp = 4/3
    fig, ((tax1, ax1, ax2), (tax2, ax3, ax4)) = plt.subplots(
                                                nrows=2, ncols=3, sharex=True,
                                                width_ratios=[0.01, 1, 1], 
                                                figsize=(size, size / asp))
    # set up line formatting 
    line_color = cmap(0.75)
    scr_line_color = cmap(0.2)
    h_line_color = cmap(0.1)
    line_alpha = 0.5
    
    # plot fR screening conditon (ax1) and fifth force (ax2)
    for i, logMvir in enumerate(fR_logMvir_range):
        # cutoff solutions once they reach the splashback radius
        R_SB = galf.get_dark_matter_parameters(M_sun * 10**logMvir)['SB']
        in_range = r <= R_SB
        for j, logfR0 in enumerate(logfR0_range):
            if fR_rs_all[i, j] < 0: # don't plot fully unscreened solutions
                continue
            
            ax1.plot(r[in_range]/fR_rs_all[i, j], fR_EoM_all[i, j][in_range], 
                     color=line_color, alpha=line_alpha)
            ax2.plot(r[in_range]/fR_rs_all[i, j], fR_a5_all[i, j][in_range], 
                     color=line_color, alpha=line_alpha)
    
    # plot sym screening condition (ax3) and fifth force (ax4)
    for i, logMvir in enumerate(sym_logMvir_range):
        # cutoff solutions once they reach the splashback radius
        R_SB = galf.get_dark_matter_parameters(M_sun * 10**logMvir)['SB'] 
        in_range = r <= R_SB
        for j, logMs in enumerate(logMs_range):
            for k, logLc in enumerate(logLc_range):
                rs = sym_rs_all[i, j, k]
                if rs is np.inf: # SSB not yet occured (fully screened)
                    continue
                elif rs == -1: # fully unscreened by field threshold
                    continue
                elif rs == -2: # fully unscreened by central laplacian threshold               
                    # calculate rs, if no central laplacian threshold
                    rs = symf.get_rs(logMs, logLc, logMvir, N_r, N_th, 
                                     threshold=sym_threshold, 
                                     unscrthreshold=sym_unscrfieldthreshold, 
                                     unscrlapthreshold=1)[th_ind]
                    
                    # load sym field
                    u, u_inf = symf.load_solution(logMs, logLc, logMvir, 
                                                  N_r, N_th)
                    sym_field_all[i, j, k] = u[:, th_ind] / u_inf
                    
                    # get sym a5
                    a5 = symf.get_a5(u, u_inf, magnitude=True)[:, th_ind]
                    sym_a5_all[i, j, k] = np.abs(a5) / np.abs(a5).max()
                                        
                    # highlight these solutions in a different colour
                    linecol = scr_line_color
                else:
                    # plot valid partially screened solutions with usual parameters
                    linecol = line_color
    
                ax3.plot(r[in_range]/rs, sym_field_all[i, j, k][in_range], 
                         c=linecol, alpha=line_alpha)
                ax4.plot(r[in_range]/rs, sym_a5_all[i, j, k][in_range], 
                         c=linecol, alpha=line_alpha)
      
    # format axes and plot threshold lines
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(1, c=h_line_color, alpha=line_alpha, ls='dashed')
        ax.set_xticks(np.arange(0, 5.0001, 2))
        ax.set_xlim([-0.01, 5.01])
        ax.set_ylim([-0.01, 1.01])
    ax1.axhline(fR_threshold, c=h_line_color, alpha=line_alpha, ls='dashed')
    ax3.axhline(sym_threshold, c=h_line_color, alpha=line_alpha, ls='dashed')
    
    # set axis labels and titles
    ax3.set_xlabel(r'$r\ /\ r_s$')
    ax4.set_xlabel(r'$r\ /\ r_s$')
    ax1.set_ylabel(r'$\delta R\ /\ 8\pi G\delta\rho/c^2$')
    ax2.set_ylabel(r'$|\mathbf{a}_{5,f_R}|\ /\ $' + 
                   r'$\mathrm{max}(|\mathbf{a}_{5,f_R}|)$')
    ax3.set_ylabel(r'$\varphi\ /\ \varphi_\infty$')
    ax4.set_ylabel(r'$|\mathbf{a}_{5,sym}|\ /\ $' + 
                   r'$\mathrm{max}(|\mathbf{a}_{5,sym}|)$')
    ax1.set_title('Screening Conditions')
    ax2.set_title('Fifth Forces')
    row_labels = [r'$f(R)$', 'Symmetron']
    for i, tax in enumerate([tax1, tax2]):
        tax.axis('off')
        tax.text(0.5, 0.5, s=row_labels[i], va='center', rotation=90,
                 fontsize=plt.rcParams['axes.titlesize'])
    
    fig.tight_layout(pad=0.1)
    
    plt.show()
            
    if savefigure is True:
        plt.savefig('Fig_2._Stacked_screening_conditions_new.png', dpi=dpi)
        print('Fig 2. saved!')
    
    fintime = time.time()
    print('Fig. 2 took {:.2f}s'.format(fintime-starttime))   


#%% Fig 3 

# Screening radius for a range of input parameters overplotted on density
def plot_figure_3(logMvir=11.5, logMs_fixed=-4.5, logLc_fixed=-1,
        logfR0_range=np.array([-6.3, -6.4, -6.5, -6.6, -7.0, -7.3, -7.6]),
        logMs_range=np.array([-4.2, -4.6, -4.8, -4.9, -5.1, -5.8, -6.1, -6.3]),
        logLc_range = np.array([-1.5, -0.9, -0.7, -0.6, -0.4, 0.3, 0.6, 0.9]),
        savefigure=False):
    """Plots figure 3 -- 2x2 figure with panels representing:
    f(R) fR0 screening surfaces       ; density colourbar
    symmetron Ms screening surfaces   ; symmetron Lc screening surfaces"""
    starttime = time.time()

    # Derived model parameters:
    Mvir = 10**logMvir * M_sun
    
    # Derived dark matter / stellar disc paramters:
    dmp = galf.get_dark_matter_parameters(Mvir)
    sdp = galf.get_stellar_disc_parameters(Mvir)
    drho = galf.get_densities(Mvir, grid.r, grid.theta, 
                              splashback_cutoff=True, total=True)['total']
    
    # load fR field screening radius
    rs_fR0 = {}
    rs_fR0_chi = {}
    for logfR0 in logfR0_range[::-1]:
        rs = fRf.get_rs(logfR0, logMvir, N_r, N_th) 
        if isinstance(rs, int):
            continue
        rs_fR0[logfR0] = rs / kpc
        rs_chi = fRf.get_rs_chi(logfR0, drho)
        rs_fR0_chi[logfR0] = np.ones_like(rs) * rs_chi / kpc                          
    
    # load Ms sym field screening radius
    rs_Ms = {}
    rs_Ms_SSB = {}
    for logMs in logMs_range:
        rs = symf.get_rs(logMs, logLc_fixed, logMvir, N_r, N_th) 
        if isinstance(rs, int) or rs is np.inf:
            continue 
        rs_Ms[logMs] = rs / kpc
        rs_SSB = symf.get_rho_SSB_rs(logMs, logLc_fixed, drho)
        rs_Ms_SSB[logMs] = rs_SSB / kpc
        
    # load Lc sym field screening radius
    rs_Lc = {}
    rs_Lc_SSB = {}
    for logLc in logLc_range:
        rs = symf.get_rs(logMs_fixed, logLc, logMvir, N_r, N_th)
        if isinstance(rs, int) or rs is np.inf:
            continue
        rs_Lc[logLc] = rs / kpc
        rs_SSB = symf.get_rho_SSB_rs(logMs_fixed, logLc, drho)
        rs_Lc_SSB[logLc] = rs_SSB / kpc
            
    # package rs solutions
    rss = [rs_fR0, rs_Ms, rs_Lc]
    rss_approx = [rs_fR0_chi, rs_Ms_SSB, rs_Lc_SSB]
    
    # calculate log density (masking invalid values from division by zero)
    logrho = np.ma.log10(drho / rho_m)
    
    # add 5% extra to radial extent so the plot fills the full semicircle
    rmaxplotted = 5* np.array([sdp['R'], 2*dmp['Rs']]) * 1.05
    
    # get extended coordinates/variables & package
    the, reL, logrhoeL = get_full_polars(grid.theta, grid.r, logrho, 
                                                 rmaxplotted=rmaxplotted[0])
    _, reR, logrhoeR = get_full_polars(grid.theta, grid.r, logrho, 
                                               rmaxplotted=rmaxplotted[1])
    re = [reL, reR]
    logrhoe = [logrhoeL, logrhoeR]
    
    rmaxplotted /= 1.05 # Remove the extra 5%
    
    # define contour levels
    levels = np.arange(np.floor(np.nanmin(logrhoeR)*2)/2, 
                       np.ceil(np.nanmax(logrhoeL)*2)/2+1e-6, 0.5)
    
    logfR0plotted = []
    logMsplotted = []
    logLcplotted = []
    paramplotted = [logfR0plotted, logMsplotted, logLcplotted]
    
    # set up figure
    asp = 1
    fig = plt.figure(figsize=(size, size / asp))
    
    # define axes locations
    dX = 0.42
    dY = 0.42
    Y1, Y2 = 0.51, 0.01
    X1L, X2L = -0.05, 0.42
    X1R, X2R = 0.205, 0.675
    dXcb, dYcb = 0.25, 0.35
    Xcb, Ycb = 0.63, 0.52
    dXlg, dYlg = 0, 0
    Xlg, Ylg = 0.985, 1.0
    # set up axes
    axfRL = fig.add_axes([X1L, Y1, dX, dY], projection='polar')
    axfRR = fig.add_axes([X1R, Y1, dX, dY], projection='polar')
    axMsL = fig.add_axes([X1L, Y2, dX, dY], projection='polar')
    axMsR = fig.add_axes([X1R, Y2, dX, dY], projection='polar')
    axLcL = fig.add_axes([X2L, Y2, dX, dY], projection='polar')
    axLcR = fig.add_axes([X2R, Y2, dX, dY], projection='polar')
    cax = fig.add_axes([Xcb, Ycb, dXcb, dYcb])
    lgax = fig.add_axes([Xlg, Ylg, dXlg, dYlg])
    axs = [axfRL, axfRR, axMsL, axMsR, axLcL, axLcR]
    
    # add titles
    axfRT = fig.add_axes([0.085, 0.95, 0.4, 0.01])
    axMsT = fig.add_axes([0.03, 0.45, 0.5, 0.01])
    axLcT = fig.add_axes([0.5, 0.45, 0.5, 0.01])
    axfRrow = fig.add_axes([-0.025, 0.44, 0.1, 0.5])
    axsymrow = fig.add_axes([-0.025, -0.115, 0.1, 0.5])
    for ax in [axfRT, axMsT, axLcT, axfRrow, axsymrow]:
        ax.axis('off')
    text_fR = r'$\log_{10}(|f_{R0}|)$'
    text_Ms1 = r'$\log_{10}(M_\mathrm{sym}/M_\mathrm{pl})$'
    text_Ms2 = r'$[\log_{{10}}(L_c/\mathrm{{kpc}}) = $' \
                + r'${:.1f}]$'.format(logLc_fixed)
    text_Lc1 = r'$\log_{10}(L_c/\mathrm{kpc})$'
    text_Lc2 = r'$[\log_{{10}}(M_\mathrm{{sym}}/M_\mathrm{{pl}}) = $' \
                + r'${:.1f}]$'.format(logMs_fixed)
    text_fRrow = r'$f(R)$'
    text_symrow = 'Symmetron'
    axfRT.text(0.5, 0.5, s=text_fR, ha='center')
    axMsT.text(0.27, 0.5, text_Ms1, ha='center')
    axMsT.text(0.74, 0.5, text_Ms2, ha='center', fontsize=10)
    axLcT.text(0.27, 0.5, text_Lc1, ha='center')
    axLcT.text(0.74, 0.5, text_Lc2, ha='center', fontsize=10)
    axfRrow.text(0.5, 0.5, s=text_fRrow, ha='center', fontsize=20, rotation=90)
    axsymrow.text(0.5, 0.5, s=text_symrow, ha='center', fontsize=20, rotation=90)
    
    # set up theta coordinate for label
    label_angle = 0.75 * np.pi
    label_ind = np.argmin(abs(the - label_angle))
    
    # plot screening surfaces
    for ind, ax in enumerate(axs):
        LR_ind = ind%2 # left/right hand side index [stellar dom., DM dom.]
        param_ind = ind//2 # parameter index [fR0, Ms, Lc]
        
        # formatting axes
        ax.set_theta_offset((-1)**LR_ind * - 0.5 * np.pi)
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticklabels([])
        ax.set_ylim([0, rmaxplotted[LR_ind]/kpc])
        ax.tick_params(axis='y', pad=-2)
    
        # plot density countour background
        im = ax.contourf(the, re[LR_ind]/kpc, logrhoe[LR_ind], 
                         cmap='inferno', levels=levels)
            
        for x, y in rss[param_ind].items():
            y_approx = rss_approx[param_ind][x]
            # plot rs if within range and not plotted already in LHS
            if np.min(y) <= rmaxplotted[LR_ind] / kpc \
                and x not in paramplotted[param_ind]:
                # extend rs to cover full 180 degrees
                ye = np.hstack((y[-1], y, y[0]))
                # plot rs and save as plotted
                line, = ax.plot(the, ye, label=str(x), 
                                color='skyblue', linestyle='--', linewidth=1.5)
                paramplotted[param_ind].append(x) 
                # annotate rs line with text box
                ax.annotate(str(x), xy=(the[label_ind], ye[label_ind]),
                        fontsize=10, rotation=180-np.rad2deg(the[label_ind]),
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.7, pad=2),
                        path_effects=[path_effects.withStroke(foreground='w',
                                                              linewidth=3)])
                
                # plot the approximate rs 
                # extend rs to cover full 180 degrees
                ye_approx = np.hstack((y_approx[-1], y_approx, y_approx[0]))
                line_approx, = ax.plot(the, ye_approx, label=str(x), 
                                       c='grey', ls='-.', lw=1.5)
                
    
    # Add colorbar for the density contourf
    cbar = plt.colorbar(im, cax=cax, pad=0)
    cbar.ax.set_ylabel(r"$\log_{10}(\rho\ /\ \bar{\rho})$")
    
    # add legend
    lgax.axis('off')
    lgax.legend([line, line_approx], ['True screening surface', r'Approximate $r_s$'])
    
    if savefigure is True:
        plt.savefig("Fig_3._Partial_Screening_new.png", dpi=dpi)
        print('Fig 3. saved!')
    
    fintime = time.time()
    print('Fig. 3 took {:.2f}s'.format(fintime-starttime))


#%% Fig 4

# Parameter space plot showing percent of mass screened against Mvir/fR0
def plot_figure_4():
    starttime = time.time()
    
    ### input params ###
    global_dX = 0.1
    # Screening model parameters:
    dfR0 = global_dX
    dMs = global_dX
    dLc = global_dX
    logMs_fixed = -4.5
    logLc_fixed = -1.0
    logfR0_range = np.arange(-8, -5+dfR0/2, dfR0)
    logMs_range = np.arange(-6.5, -3+dMs/2, dMs)
    logLc_range = np.arange(-3, 2+dLc/2, dLc) # Larger than 2 --> no SSB
    # NFW profile input paramters:
    dMvir = global_dX
    logMvir_range = np.arange(10, 13.5+dMvir/2, dMvir)
    # Screening radius parameters:
    fR_threshold = 0.9
    sym_threshold = 0.1
    sym_unscrthreshold = 1e-3
    sym_unscrlapthreshold = 1e-1
    ####################
    
    included_components = ['SD', 'DM']#, 'total']
    
    # set up grid structure
    th_ind = grid.disc_idx
    r = grid.r[:, th_ind]
    
    # calculate pms
    fR_pms = {}
    fR_chi_pms = {}
    for comp in included_components:
        fR_pms[comp] = np.zeros([logfR0_range.size, logMvir_range.size])
        fR_chi_pms[comp] = np.zeros_like(fR_pms[comp])
    
    for j, logMvir in enumerate(logMvir_range):
        Mvir = M_sun * 10**(logMvir)
        rhos = galf.get_densities(Mvir, grid.r, grid.theta, 
                                  splashback_cutoff=True, total=True)       
        for i, logfR0 in enumerate(logfR0_range):
            # calculate true and approx rs solutions
            rs = fRf.get_rs(logfR0, logMvir, N_r, N_th, threshold=fR_threshold, 
                            unscrthreshold=fR_unscrthreshold)
            rs_chi = fRf.get_rs_chi(logfR0, rhos['total'])
    
            # calculate % of mass component within rs
            for comp in included_components:
                fR_pms[comp][i, j] = mass_enclosed(grid, rhos[comp], rs) 
                fR_chi_pms[comp][i, j] = mass_enclosed(grid, rhos[comp], rs_chi)
    
                
    Ms_pms = {}
    Ms_SSB_pms = {}
    for comp in included_components:
        Ms_pms[comp] = np.zeros([logMs_range.size, logMvir_range.size])
        Ms_SSB_pms[comp] = np.zeros_like(Ms_pms[comp])
        
    for j, logMvir in enumerate(logMvir_range):
        Mvir = M_sun * 10**(logMvir)
        rhos = galf.get_densities(Mvir, grid.r, grid.theta, 
                                  splashback_cutoff=True, total=True)     
        for i, logMs in enumerate(logMs_range):
            # calculate true and approx rs solutions
            rs = symf.get_rs(logMs, logLc_fixed, logMvir, N_r, N_th, 
                             threshold=sym_threshold, 
                             unscrthreshold=sym_unscrthreshold, 
                             unscrlapthreshold=sym_unscrlapthreshold)
            rs_SSB = symf.get_rho_SSB_rs(logMs, logLc_fixed, rhos['total'])
            
            # calculate % of mass component within rs
            for comp in included_components:
                Ms_pms[comp][i, j] = mass_enclosed(grid, rhos[comp], rs) 
                Ms_SSB_pms[comp][i, j] = mass_enclosed(grid, rhos[comp], rs_SSB) 
                
    Lc_pms = {}
    Lc_SSB_pms = {}
    for comp in included_components:
        Lc_pms[comp] = np.zeros([logLc_range.size, logMvir_range.size])
        Lc_SSB_pms[comp] = np.zeros_like(Lc_pms[comp])
        
    for j, logMvir in enumerate(logMvir_range):
        Mvir = M_sun * 10**(logMvir)
        rhos = galf.get_densities(Mvir, grid.r, grid.theta, 
                                  splashback_cutoff=True, total=True) 
        for i, logLc in enumerate(logLc_range):
            # calculate true and approx rs solutions     
            rs = symf.get_rs(logMs_fixed, logLc, logMvir, N_r, N_th,
                             threshold=sym_threshold, unscrthreshold=sym_unscrthreshold, 
                             unscrlapthreshold=sym_unscrlapthreshold)
            rs_SSB = symf.get_rho_SSB_rs(logMs_fixed, logLc, rhos['total'])
            
            # calculate % of mass component within rs
            for comp in included_components:
                Lc_pms[comp][i, j] = mass_enclosed(grid, rhos[comp], rs)
                Lc_SSB_pms[comp][i, j] = mass_enclosed(grid, rhos[comp], rs_SSB)
    
    # get binary screening boundary (with extended Mvir to cover whole plot)
    logMvir_extened = np.array([logMvir_range[0] - dMvir] + list(logMvir_range) 
                               + [logMvir_range[-1] + dMvir])
    logfR0_crit_range = fRf.logfR0_crit_binary_screening(logMvir_extened)
    logMs_crit_range = symf.logMs_crit_binary_screening(logMvir_extened)
    
    ### Plotting Fig 4 ###
    
    # set shading style
    shading = 'nearest' # use 'gouraud' for interpolation
    Mvir_edgelen = 0
    fR_edgelen = 0
    Ms_edgelen = 0
    Lc_edgelen= 0
    if shading == 'nearest':
        Mvir_edgelen += dMvir/2
        fR_edgelen += dfR0/2
        Ms_edgelen += dMs/2
        Lc_edgelen += dLc/2
    
    cmap.set_under(cmap(0)) # corresponds to fully unscreened solutions
    
    asp = 1
    fig = plt.figure(figsize=(size, size/asp), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=len(included_components)+2, 
                          height_ratios=[1, 1, 1], 
                          width_ratios=[0.01]+[1]*len(included_components)+[0.1])
    
    row_labels = [r'$f_R$', 'Symmetron']
    ylabels = [r'$\log_{10}(|f_{R0}|)$', 
               r'$\log_{10}(M_\mathrm{sym}/M_\mathrm{pl})$', 
               r'$\log_{10}(L_c/\mathrm{kpc})$']
    # add subplots to the grid
    axes = []
    for i in range(gs.nrows):
        row_axes = []
        for j in range(gs.ncols - 2):
            j += 1
            ax = fig.add_subplot(gs[i, j])
            if j == 1:
                ax.set_ylabel(ylabels[i], labelpad=0)
            else:
                ax.set_yticklabels([])
            if i == gs.nrows - 1:
                ax.set_xlabel(r'$\log_{10}(M_\mathrm{vir}/M_\odot)$', labelpad=0)
            else:
                ax.set_xticklabels([])
            ax.set_xlim([logMvir_range.min() - Mvir_edgelen, 
                         logMvir_range.max() + Mvir_edgelen])
            row_axes.append(ax)
        axes.append(row_axes)
    axes = np.array(axes)
    
    # add colorbar subplot to the top of the grid
    cbar_ax = fig.add_subplot(gs[:, -1])
    
    # add row labels to LHS of the grid
    fR_row_ax = fig.add_subplot(gs[0, 0])
    sym_row_ax = fig.add_subplot(gs[1:3, 0])
    for i, ax in enumerate([fR_row_ax, sym_row_ax]):
        ax.axis('off')
        ax.text(0.5, 0.4, s=row_labels[i], ha='center', fontsize=20, rotation=90)
    
    # line colours/styles for 
    # [50% screened, binary screening, 50% screened (approximate rs)]
    line_colors = ['skyblue', 'grey', 'grey']
    line_styles = ['--', ':', '-.',] 
    line_width = 2
    
    column_titles = {'DM': 'Halo Mass', 'SD': 'Disc Mass'} 
       
    # convert screened fraction to screened %
    for comp in included_components:
        for pms in [fR_pms, fR_chi_pms, Ms_pms, Ms_SSB_pms, Lc_pms, Lc_SSB_pms]:
            pms[comp] *= 100
    
    # plot the data
    for col_ind, comp in enumerate(included_components):
        # fR plot
        row_ind = 0
        ax = axes[row_ind, col_ind]
        im = ax.pcolormesh(logMvir_range, logfR0_range, fR_pms[comp], 
                           shading=shading, vmin=0, vmax=100, cmap=cmap)
        # plot the line where >50% of mass is screened (true and approx)
        ax.contour(logMvir_range, logfR0_range, fR_pms[comp], levels=[50], 
                   colors=line_colors[0], linestyles=line_styles[0], linewidths=line_width)
        ax.contour(logMvir_range, logfR0_range, fR_chi_pms[comp], levels=[50],
                    colors=line_colors[2], linestyles=line_styles[2], linewidths=line_width)
        # plot the line where potential at virial radius > critical potential
        ax.plot(logMvir_extened, logfR0_crit_range, 
                color=line_colors[1], linestyle=line_styles[1], linewidth=line_width)
        # plot the dividing line between fully screened and partial screened
        bool_boundary_divider((fR_pms[comp] < 0), logMvir_range, logfR0_range, 
                              ax, 'w', linewidth=1)
        ax.set_ylim([logfR0_range.min() - fR_edgelen, logfR0_range.max() + fR_edgelen])
        
        ax.set_title(column_titles[comp])
    
        # sym Ms plot
        row_ind += 1
        ax = axes[row_ind, col_ind]
        ax.pcolormesh(logMvir_range, logMs_range, Ms_pms[comp], 
                      shading=shading, vmin=0, vmax=100, cmap=cmap)  
        # plot the line where >50% of mass is screened (true and approx)
        ax.contour(logMvir_range, logMs_range, Ms_pms[comp], levels=[50],
                   colors=line_colors[0], linestyles=line_styles[0], linewidths=line_width)
        ax.contour(logMvir_range, logMs_range, Ms_SSB_pms[comp], levels=[50], 
                   colors=line_colors[2], linestyles=line_styles[2], linewidths=line_width)
        # plot the line where potential at virial radius > critical potential
        ax.plot(logMvir_extened, logMs_crit_range, 
                color=line_colors[1], linestyle=line_styles[1], linewidth=line_width)
        # plot the dividing line between fully screened and partial screened
        bool_boundary_divider((Ms_pms[comp] < 0), logMvir_range, logMs_range, 
                              ax, 'w', linewidth=1)
        ax.set_ylim([logMs_range.min() - Ms_edgelen, logMs_range.max() + Ms_edgelen])
    
    
        # sym Lc plot
        row_ind += 1
        ax = axes[row_ind, col_ind]
        ax.pcolormesh(logMvir_range, logLc_range, Lc_pms[comp], 
                      shading=shading, vmin=0, vmax=100, cmap=cmap)
        # plot the line where >50% of mass is screened (true and approx)
        ax.contour(logMvir_range, logLc_range, Lc_pms[comp], levels=[50],
                   colors=line_colors[0], linestyles=line_styles[0], linewidths=line_width)
        ax.contour(logMvir_range, logLc_range, Lc_SSB_pms[comp], levels=[50], 
                   colors=line_colors[2], linestyles=line_styles[2], linewidths=line_width)
        # plot the dividing line between fully screened and partial screened
        bool_boundary_divider((Lc_pms[comp] < 0), logMvir_range, logLc_range, 
                              ax, 'w', linewidth=1)
        ax.set_ylim([logLc_range.min() - Lc_edgelen, logLc_range.max() + Lc_edgelen])
    
    # add a horizontal colorbar
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticklocation='right')
    cbar_ax.set_ylabel('Screened Mass Fraction [\%]', rotation=270,
                       fontsize=plt.rcParams['axes.titlesize'], labelpad=10)
    
    ### Adding labels to f(R) DM plot ###
    ax = axes[0][1]
    rot_deg = 31
    rot_rad = rot_deg * np.pi / 180
    srot = np.sin(rot_rad)
    crot = np.cos(rot_rad)
    labelargs = dict(rotation=rot_deg, fontsize=11, fontweight='bold')
    
    # fully screened region
    label = r"$\mathbf{Fully\ Unscreened}$"
    ax.annotate(label, (10.25, -6.55), c='w', **labelargs)
    x1 = 10.5
    y1 = -7.0
    x2 = 12.6
    y2 = -5.6
    L = 0.5
    arrargs = dict(c='w', arrowprops=dict(arrowstyle="->", ec='w', lw=1))
    ax.annotate("", (x1 - L * srot, y1 + L * crot), (x1, y1), **arrargs)
    ax.annotate("", (x2 - L * srot, y2 + L * crot), (x2, y2), **arrargs)
    
    # 50% screened line
    label = r"$\mathbf{50\%\ Screened}$"
    ax.annotate(label, (11.7, -7.7), c='skyblue', **labelargs)
    
    # binary line
    label = r"$\mathbf{Binary\ Threshold}$"
    ax.annotate(label, (11.2, -7.2), c='grey', **labelargs)
    
    # 50% approximate line
    label = r"$\mathbf{50\%\ Screened;\ r_s\ Approx.}$"
    ax.annotate(label, (10.1, -7.55), c='grey', **labelargs)
    x1 = 10.9
    y1 = -7.1
    L = 0.6
    arrargs = dict(c='w', arrowprops=dict(arrowstyle="->", ec='grey', lw=1.2))
    ax.annotate("", (x1 + L * srot, y1 - L * crot), (x1, y1), **arrargs)
    ### ###
    
    fig.get_layout_engine().set(h_pad=-1)
    plt.show()
    
    if savefigure is True:
        plt.savefig("Fig_4._Mass_Screened_new.png", dpi=dpi)
        print('Fig 4. saved!')
    
    fintime = time.time()
    print('Fig. 4 took {:.2f}s'.format(fintime-starttime))    

#%% Fig 5
# plot showing the screening radius in a the MW for a range of fR0 / sym params

def plot_figure_5():
    starttime = time.time()
    
    ### input params ###
    # Screening model parameters:
    dfR0 = 0.05
    dMs, dLc = 0.1, 0.1
    logfR0_range = np.arange(-8, -6.2, dfR0)
    logfR0_range = np.append(logfR0_range, np.arange(-6.2, -5.8, 0.01))
    logfR0_range = np.append(logfR0_range, np.arange(-5.8, -5+0.01, dfR0))
    logfR0_range = logfR0_range
    logMs_range = np.arange(-6.5, -3+0.01, dMs)
    logLc_range = np.arange(-3, 3+0.01, dLc)
    # NFW profile input paramters:
    Mvir = 1.5e12 * M_sun   # MW mass
    logMvir = np.log10(Mvir / M_sun)
    logMvir = np.log10(1.5e12)
    # Screening condition inputs:
    fR_threshold = 0.9
    fR_unscrthreshold = 1e-3
    sym_threshold = 0.1
    sym_unscrthreshold = 1e-3
    sym_unscrlapthreshold = 1e-1
    ####################
    
    # set up grid (lower resolution in theta since only plotting along disc plane)
    grid5 = fR2DSolver(N_r=N_r, N_th=int(101), r_min=r_min, r_max=r_max)
    th_ind = grid5.disc_idx
    r = grid5.r[:, th_ind]
    
    drho = galf.get_densities(Mvir, grid5.r, grid5.theta, total=True)['total']
    
    rs = {}
    rs_approx = {}
    for logfR0 in logfR0_range:
        rs[logfR0] = fRf.get_rs(logfR0, logMvir, N_r, N_th=101,
                                threshold=fR_threshold, 
                                unscrthreshold=fR_unscrthreshold)
        if isinstance(rs[logfR0], np.ndarray):
            rs[logfR0] = rs[logfR0][th_ind] / kpc   
        rs_approx[logfR0] = fRf.get_rs_chi(logfR0, drho) / kpc
    
    rss = np.zeros([logMs_range.size, logLc_range.size])
    rss_approx = np.zeros_like(rss)
    for i, logMs in enumerate(logMs_range):
        for j, logLc in enumerate(logLc_range):
            temp_rss = symf.get_rs(logMs=logMs, logLc=logLc, logMvir=logMvir, 
                                    N_r=N_r, N_th=101, threshold=sym_threshold, 
                                    unscrthreshold=sym_unscrthreshold, 
                                    unscrlapthreshold=sym_unscrlapthreshold)
            temp_approx = symf.get_rho_SSB_rs(logMs, logLc, drho)
            if isinstance(temp_rss, np.ndarray):
                rss[i, j] = temp_rss[th_ind] / kpc
            else:
                rss[i, j] = temp_rss
            if isinstance(temp_approx, np.ndarray):
                rss_approx[i, j] = temp_approx[th_ind] / kpc
            else:
                rss_approx[i, j] = temp_approx
                
    logMs_crit = symf.logMs_crit_binary_screening(logMvir)
    logfR0_crit = fRf.logfR0_crit_binary_screening(logMvir)
    
    linecolor = 'skyblue'
    linewidth = 2
    fontsize = 11
    asp = 1
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(size, size/asp), 
                                   height_ratios=[0.6, 1], constrained_layout=False)
    # inset axis
    axins = ax1.inset_axes([0.52, 0.3, 0.47, 0.68])
    
    # plot f(R) screening radii
    for ax in [ax1, axins]:
        line, = ax.plot(rs.keys(), rs.values(), 
                        color='skyblue', linewidth=3, linestyle='--')
        # plot f(R) approximate screening radii
        approx_line, = ax.plot(rs_approx.keys(), rs_approx.values(), 
                               color='grey', linewidth=2, linestyle='-.')
        # MW-Solar System seperation
        ax.axhline(8, color='k', linewidth=1)    
        
    ax1.legend([line, approx_line], [r'True $r_s$', r'Approx. $r_s$'],
               loc='lower left', bbox_to_anchor=(0.68, 0.65), fontsize=fontsize)
    
    # Binary screening condition
    ax1.axvline(logfR0_crit, color='grey', linestyle=':', linewidth=2)
    # plot text labels
    ax1.text(0.05, 0.05, r'$\bm{r_s=8\,\mathrm{kpc}}$', transform=ax1.transAxes, 
             fontsize=fontsize, color='k')
    ax1.text(0.48, 0.64, r'$\mathbf{Binary\ Fully}$' '\n' r'$\mathbf{Unscreened}$', 
             transform=ax1.transAxes,
             fontsize=fontsize, color='grey', rotation=90, ha='center', va='center')
    ax1.text(0.395, 0.64, r'$\mathbf{Binary\ Fully}$' '\n' '$\mathbf{Screened}$', 
             transform=ax1.transAxes, 
             fontsize=fontsize, color='grey', rotation=90, ha='center', va='center')
    # Formatting
    ax1.set_xlabel('$\log_{10} f_{R0}$')
    ax1.set_ylabel('$r_s\ /\ \mathrm{kpc}$')
    ax1.set_xlim(logfR0_range.min(), logfR0_range.max())
    ax1.set_ylim(bottom=0)
    # insert formatting
    axins.set_xlim(-6.2, -5.8)
    axins.set_ylim(0, 21)
    axins.set_yticklabels([])
    ax1.indicate_inset_zoom(axins)
    
    ## plot sym screening radii
    im = ax2.pcolormesh(logLc_range, logMs_range, rss, vmin=0, vmax=300,
                       shading='auto', cmap=cmap)
    # plot fully screened boundaries
    bool_boundary_divider((rss<0), x=logLc_range, y=logMs_range, 
                              ax=ax2, linecolor='w', linewidth=1)
    # MW-Solar System seperation
    ax2.contour(logLc_range, logMs_range, rss,
               levels=[8], colors=linecolor, linestyles='--', linewidths=3)
    # MW-Solar System seperation FROM APPROXIMATION
    ax2.contour(logLc_range, logMs_range, rss_approx,
               levels=[8], colors='grey', linestyles='-.', linewidths=2)
    # Binary screening condition
    ax2.axhline(logMs_crit, color='grey', linestyle=':', linewidth=2)
    
    # plot text labels
    ax2.text(0.04, 0.52, r'$\mathbf{Fully\ Unscreened}$', 
             transform=ax2.transAxes, color='w', fontsize=fontsize, rotation=40)
    ax2.text(0.50, 0.15, r'$\mathbf{Fully\ Screened}$', 
             transform=ax2.transAxes, color='k', fontsize=fontsize, rotation=49)
    ax2.text(0.8, 0.05, r'$\bm{\rho_\mathrm{SSB} < \bar{\rho}}$', 
             transform=ax2.transAxes, color='k', fontsize=fontsize)
    ax2.text(0.12, 0.17, r'$\bm{r_s=8\,\mathrm{kpc}}$', 
             transform=ax2.transAxes, color=linecolor, fontsize=fontsize, rotation=49)
    ax2.text(0.03, 0.17, r'$\bm{\mathrm{Approx.}\ r_s=8\,\mathrm{kpc}}$', 
             transform=ax2.transAxes, color='grey', fontsize=fontsize, rotation=49)
    ax2.text(0.05, 0.965, r'$\mathbf{Binary\ Fully\ Unscreened}$', 
             transform=ax2.transAxes, color='grey', fontsize=fontsize)
    ax2.text(0.075, 0.91, r'$\mathbf{Binary\ Fully\ Screened}$', 
             transform=ax2.transAxes, color='grey', fontsize=fontsize)
    
    # plot annotation arrows
    rot_deg = 45
    rot_rad = rot_deg * np.pi / 180
    srot = np.sin(rot_rad)
    crot = np.cos(rot_rad)
    x1 = -2.43
    y1 = -5.07
    x2 = -0.93
    y2 = -3.97
    L = 0.5
    arrargs = dict(c='w', arrowprops=dict(arrowstyle="->", ec='w', lw=1.2))
    ax2.annotate("", (x1 - L * srot, y1 + L * crot), (x1, y1), **arrargs)
    ax2.annotate("", (x2 - L * srot, y2 + L * crot), (x2, y2), **arrargs)
    
    # plot colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cbar = plt.colorbar(im, cax=cax, extend='max')
    # Formatting
    ax2.set_xlabel('$\log_{10} (L_c / \mathrm{kpc})$')
    ax2.set_ylabel('$\log_{10} (M_\mathrm{sym} / M_\mathrm{pl})$')
    cbar.ax.set_ylabel(r"$r_s\ /\ \mathrm{kpc}$")
    ax2.set_xlim([logLc_range.min() - dLc/2, logLc_range.max() + dLc/2])
    ax2.set_ylim([logMs_range.min() - dMs/2, logMs_range.max() + dMs/2])
    fig.tight_layout(pad=0.3)
    
    plt.show()
    
    if savefigure is True:
        plt.savefig('Fig 5. MW Screening Radius (new).png')
        print('Fig 5. saved!')
    
    fintime = time.time()
    print('Fig. 5 took {:.2f}s'.format(fintime-starttime))  

#%% Timings

if __name__ == '__main__':
    tstart = time.time()
    
    """
    plot_figure_1(logMvir=11.5, logfR0=-6.4, logMs=-4.5, logLc=-1, grid=grid,
                  N_stellar_scale_length_plotted=5, N_grid_lines_plotted=5,
                  savefigure=False)
    #"""
    
    """
    plot_figure_2(dfR0=0.2, fR_dMvir=0.2, dMs=0.5, dLc=0.5, sym_dMvir=0.5, 
                      grid=grid, fR_threshold = 0.9, fR_unscrthreshold = 1e-3,
                      sym_threshold = 0.1, sym_unscrfieldthreshold = 1e-3,
                      sym_unscrlapthreshold = 1e-1)
    #"""
    
    #"""
    plot_figure_3(logMvir=11.5, logMs_fixed=-4.5, logLc_fixed=-1,
            logfR0_range=np.array([-6.3, -6.4, -6.5, -6.6, -7.0, -7.3, -7.6]),
            logMs_range=np.array([-4.2, -4.6, -4.8, -4.9, -5.1, -5.8, -6.1, -6.3]),
            logLc_range = np.array([-1.5, -0.9, -0.7, -0.6, -0.4, 0.3, 0.6, 0.9]),
            savefigure=False)
    #"""
    tfinish = time.time()
    print('Total Time Taken:', tfinish-tstart)
