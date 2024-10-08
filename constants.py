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

# Cosmological parameters
h = 0.7 # Hubble factor
H0 = h * 100.0 * 1000.0 / Mpc # Hubble constant
omega_m = 0.3 # matter density parameter
rho_c = 3.0 * H0**2 / (8.0 * np.pi * G) # critical density
rho_m = rho_c * omega_m # mean cosmic matter density
R0 = 3 * omega_m * H0**2 * (1 + 4 * (1 - omega_m) / omega_m) / c**2 # background curvature

