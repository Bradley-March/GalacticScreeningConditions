# -*- coding: utf-8 -*-
"""
Created: June 2022
Author: Bradley March
"""
#%% Preamble 
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc # https://pypi.org/project/pynverse/ (0.1.4.6)
from constants import pc, kpc, Mpc, h, M_sun, rho_c, rho_m

# define galaxy profile constants
r_min, r_max = 50 * pc, 10 * Mpc # inner/outer grid cutoff
splashback_factor = 2.2 # outer density cutoff
delta = 200 # virial density contrast, i.e. Mvir = M200c

#%% Stellar-Halo Mass Relations

def virial_mass_to_stellar_mass(virial_mass, logM1=11.590, N=0.0351, 
                                beta=1.376, gamma=0.608):
    """Stellar-Halo Mass Relation (SHMR). 
    [Equation 2 in Moster, Naab and White (2012), parameters from Table 3.]"""
    M1 = M_sun * 10**(logM1)
    numerator = 2 * N * virial_mass
    denomenator = (virial_mass / M1)**(-beta) + (virial_mass / M1)**(gamma)
    stellar_mass =  numerator / denomenator 
    return stellar_mass

def stellar_mass_to_virial_mass(stellar_mass, logM1=11.590, N=0.0351, 
                                beta=1.376, gamma=0.608):
    """Inverse of SHMR."""
    with np.errstate(all='ignore'):
        inverse_virial_mass2stellar_mass = inversefunc(
                                                virial_mass_to_stellar_mass, 
                                                args=(logM1, N, beta, gamma), 
                                                domain=[M_sun, 1e30 * M_sun])
        virial_mass = inverse_virial_mass2stellar_mass(stellar_mass)
        return virial_mass

#%% Dark matter parameter pipeline functions

def calc_concentration_parameter(virial_mass):
    """Empirical mass-concentration relation.
    [Equation 8 of Dutton and Maccio (2014).]"""
    concentration_parameter = 10**(0.905 - 0.101*np.log10(virial_mass * 
                                                          h/(M_sun * 1e12)))
    return concentration_parameter

def calc_virial_radius(virial_mass):
    """From definition of virial mass."""
    virial_radius = (3 * virial_mass / (4 * np.pi * delta * rho_c))**(1/3)
    return virial_radius

def calc_virial_scale_radius(virial_radius, concentration_parameter):
    """From definition of concentration parameter."""
    virial_scale_radius = virial_radius / concentration_parameter
    return virial_scale_radius

def calc_virial_normalisation(virial_mass, virial_radius, 
                              concentration_parameter):
    """Reversing the relation for the enclosed mass at a given radius in an
    NFW profile to find the density normalisation."""
    virial_scale_radius = calc_virial_scale_radius(virial_radius, 
                                                   concentration_parameter)
    denom = np.log(1 + concentration_parameter) - (concentration_parameter /
                                               (1 + concentration_parameter))
    denom = 4 * np.pi * virial_scale_radius**3 * denom
    virial_normalisation = virial_mass / denom
    return virial_normalisation
    
#%% Stellar parameter pipeline functions

def calc_stellar_scale_length(stellar_mass, alpha=0.14, beta=0.39, gamma=0.10, 
                              M_0=3.98e10*M_sun):
    """Calculate scale length by passing from mass to half-light radius 
    [Equation 18 in Shen et al. 2003, parameters from Table 1.]
    and then analytically relating the half-light radius to the radius where 
    half the stellar mass is enclosed."""
    # calculate half-light radius 
    R_hl = gamma * (stellar_mass / M_sun)**(alpha) \
                   * (1 + stellar_mass / M_0)**(beta - alpha)
    # convert to stellar radial scale length
    stellar_scale_length = 0.595824 * R_hl * kpc
    return stellar_scale_length

def calc_stellar_normalisation(stellar_mass, stellar_scale_length):
    """Reversing the total stellar mass defintion to find the (area) density 
    normalisation."""
    denom = 2 * np.pi * stellar_scale_length**2
    stellar_normalisation = stellar_mass / denom
    return stellar_normalisation

def calc_stellar_scale_height(stellar_scale_length):
    """Empirical relation between scale length and height.
    [Equation 1 of Bershady et al. (2010).]"""
    exponent = 0.367 * np.log10(stellar_scale_length / kpc) + 0.708 
    stellar_scale_height = stellar_scale_length * 10**(-exponent)
    return stellar_scale_height

#%% Calculate density profiles from parameters

def calc_dark_matter_desnity(rgrid, virial_normalisation, virial_scale_radius):
    """Calculate the dark matter density as an NFW profile.
    [Navarro, Frenk and White (1995).]"""
    x = rgrid / virial_scale_radius
    dark_matter_density = virial_normalisation / (x * (1 + x)**2)
    return dark_matter_density

def calc_stellar_disc_density(rgrid, thgrid, stellar_normalisation, 
                              stellar_scale_length, stellar_scale_height):   
    """Calculate the stellar disc density as a double-exponential profile.
    [Binney and Tremaine (2008).]"""
    # set up polar coordinates
    zgrid = np.abs(rgrid * np.cos(thgrid))
    Rgrid = rgrid * np.sin(thgrid)
    # calculate density 
    prefactor = stellar_normalisation / (2 * stellar_scale_height) 
    expR = np.exp(-Rgrid / stellar_scale_length)
    expz = np.exp(-zgrid / stellar_scale_height)
    stellar_disc_density = prefactor * expR * expz
    return stellar_disc_density

#%% Derive density profile parameters from input virial_mass

def get_dark_matter_parameters(virial_mass):
    """Runs the pipeline of empirical and analytic relations to derive all 
    dark matter density profile parameters."""
    concentration_parameter = calc_concentration_parameter(virial_mass)
    virial_radius = calc_virial_radius(virial_mass)
    virial_scale_radius = calc_virial_scale_radius(virial_radius,
                                                   concentration_parameter)
    virial_normalisation = calc_virial_normalisation(virial_mass, 
                                                     virial_radius, 
                                                     concentration_parameter)

    # create dictionary for Dark Matter Parameters 
    DMP_dict = {'M': virial_mass,
                'c': concentration_parameter,
                'R': virial_radius,
                'norm': virial_normalisation,
                'Rs': virial_scale_radius,
                'SB': splashback_factor * virial_radius}
    
    return DMP_dict

def get_stellar_disc_parameters(virial_mass):
    """Runs the pipeline of empirical and analytic relations to derive all 
    stellar disc density profile parameters."""
    stellar_mass = virial_mass_to_stellar_mass(virial_mass)
    stellar_scale_length = calc_stellar_scale_length(stellar_mass)
    stellar_normalisation = calc_stellar_normalisation(stellar_mass, 
                                                       stellar_scale_length)
    stellar_scale_height = calc_stellar_scale_height(stellar_scale_length)

    # create dictionary for Stellar Disc Parameters 
    SDP_dict = {'m': stellar_mass,
               'R': stellar_scale_length,
               'z': stellar_scale_height,
               'norm': stellar_normalisation}

    return SDP_dict

#%% Density pipeline with virial_mass as an input

def get_dark_matter_density(virial_mass, rgrid, splashback_cutoff=True):
    """Calculates the dark matter density profile for a given mass, assuming
    mean values on all empirical relations."""
    DMP = get_dark_matter_parameters(virial_mass)
    dark_matter_density = calc_dark_matter_desnity(rgrid, 
                                                   DMP['norm'], DMP['Rs'])
    if splashback_cutoff is True: # cut off density at SB radius
        dark_matter_density[rgrid >= DMP['SB']] = 0 
    return dark_matter_density

def get_stellar_disc_density(virial_mass, rgrid, thgrid, 
                             splashback_cutoff=True):
    """Calculates the stellar disc density profile for a given mass, assuming
    mean values on all empirical relations."""
    SDP = get_stellar_disc_parameters(virial_mass)
    stellar_disc_density = calc_stellar_disc_density(rgrid, thgrid, 
                                             SDP['norm'], SDP['R'], SDP['z'])
    if splashback_cutoff is True: # cut off density at SB radius
        DMP = get_dark_matter_parameters(virial_mass)
        stellar_disc_density[rgrid >= DMP['SB']] = 0
    return stellar_disc_density

def get_densities(virial_mass, rgrid, thgrid, splashback_cutoff=True, 
                  total=False):
    """Calculates all density components for a given mass, assuming mean 
    values on all empirical relations."""
    rho_DM = get_dark_matter_density(virial_mass, rgrid, splashback_cutoff)
    rho_SD = get_stellar_disc_density(virial_mass, rgrid, thgrid, 
                                      splashback_cutoff)
    
    rhos = {'DM': rho_DM,
            'SD': rho_SD}
    
    if total is True:
        rhos['total'] = np.zeros_like(rhos['DM'])
        for component, rho in rhos.items():
            if component != 'total':
                rhos['total'] += rhos[component] 
    return rhos