#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical constants, including cosmological parameters, all in SI units.

Created: Dec 2023
Author: Bradley March
"""
import numpy as np
from scipy.constants import c, G, parsec as pc

# Distance measures (parsecs)
kpc, Mpc = pc * 1e3, pc * 1e6

# Solar parameters
M_sun = 1.9885e+30 # Solar mass 
logM_sun = np.log10(M_sun)
L_sun = 3.846e+26 # Solar luminosity 
Mag_sun = 4.83 # Solar absolute magnitude

# Planck units
M_pl = 2.176434e-08
L_pl = 1.616255e-25

# Cosmological parameters
h = 0.7 # Hubble factor
H0 = h * 100.0 * 1000.0 / Mpc # Hubble constant
omega_m = 0.3 # matter density parameter
omega_L = 1 - omega_m # cosmolgical constant density parameter
rho_c = 3.0 * H0**2 / (8.0 * np.pi * G) # critical density
rho_m = rho_c * omega_m # mean cosmic matter density
rho_L = rho_c * omega_L # cosmological constant density
R0 = 3 * omega_m * H0**2 * (1 + 4 * (1 - omega_m) / omega_m) / c**2 # background curvature

