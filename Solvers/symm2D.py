#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D (spherical r and theta) symmetron solver.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np
import h5py
from time import time
from scipy.constants import G, c
from Solvers.utils import calc_rho_mean


def L_op(phi, p, q):
    """Evaluate L operator."""
    L = phi * phi * phi + p * phi + q
    return L


def dLdphi_op(phi, p):
    """Evaluate L operator."""
    dLdphi = 3 * phi**2 + p
    return dLdphi


class Symm2DSolver:
    """2D symmetron solver."""

    def __init__(self, r_min, r_max, N_r, N_th):
        # check argument types
        if type(N_r) not in [int, np.int32, np.int64]:
            raise TypeError("Need integer N_r")
        if type(N_th) not in [int, np.int32, np.int64]:
            raise TypeError("Need integer N_th")
        if type(r_min) not in [float, np.float32, np.float64]:
            raise TypeError("Need float r_min")
        if type(r_max) not in [float, np.float32, np.float64]:
            raise TypeError("Need float r_max")

        # ensure N_radial even and N_angular odd
        if (N_r % 2) or not (N_th % 2):
            raise ValueError("Need even N_r and odd N_th.")

        # save arguments as attributes
        self.r_min = r_min
        self.r_max = r_max
        self.N_r = N_r
        self.N_th = N_th

        # index of disc-plane (i.e. x[:, idx] evaluates x along disc-plane)
        self.disc_idx = (N_th - 1) // 2

        # set up radial and angular cells
        q_min = np.log(r_min)
        q_max = np.log(r_max)
        q_edges = np.linspace(q_min, q_max, N_r + 1)
        q_cen = 0.5 * (q_edges[1:] + q_edges[:-1])
        ct_min = -1
        ct_max = 1
        ct_edges = np.linspace(ct_min, ct_max, N_th + 1)
        ct_cen = 0.5 * (ct_edges[1:] + ct_edges[:-1])
        self.hq = np.diff(q_cen)[0]
        self.hct = np.diff(ct_cen)[0]

        # save some useful coordinate grids
        r_cen = np.exp(q_cen)
        r_edges = np.exp(q_edges)
        th_cen = np.arccos(ct_cen)
        st2_edges = 1 - ct_edges**2
        r, theta = np.meshgrid(r_cen, th_cen, indexing='ij')
        x = r * np.sin(theta)
        y = np.zeros_like(x)
        z = r * np.cos(theta)
        rout, stout2 = np.meshgrid(r_edges[1:], st2_edges[1:], indexing='ij')
        rin, stin2 = np.meshgrid(r_edges[:-1], st2_edges[:-1], indexing='ij')
        self.r = r
        self.theta = theta
        self.rout = rout
        self.rin = rin
        self.stout2 = stout2
        self.stin2 = stin2
        self.pos = np.stack((x, y, z), axis=2)

        # assign red/black cells; first cell (0, 0) is red
        self.black = (np.indices((N_r, N_th)).sum(axis=0) % 2).astype(bool)
        self.red = ~self.black

        # flag to indicate density hasn't been provided yet
        self.DensityFlag = False

        return

    def set_density(self, rho):
        """Provide density for symmetron solution."""
        # check argument types
        if type(rho) is not np.ndarray:
            raise TypeError("Need numpy array for rho")
        if rho.dtype != np.float64:
            raise TypeError("rho should be np.float64, not", rho.dtype)
        if rho.shape != (self.N_r, self.N_th):
            raise ValueError("rho has wrong shape: should be (N_r, N_th)")

        # save density and switch flag
        self.rho = rho
        self.DensityFlag = True

        return

    def solve(self, M_symm, l0_symm, verbose=False, tol=1e-8,
              imin=100, imax=500000, set_zero_region=True):
        """Solve symmetron EOM for given density profile."""
        # check argument types
        if type(M_symm) not in [float, np.float32, np.float64]:
            raise TypeError("Need float M_symm")
        if type(l0_symm) not in [float, np.float32, np.float64]:
            raise TypeError("Need float l0_symm")
      

        # start timer
        t0 = time()

        # compute critical density and mean matter density
        rho_SSB = (c**2 / (16 * np.pi * G * l0_symm**2)) * M_symm**2
        rho_mean = calc_rho_mean(h=0.7, omega_m=0.3)

        # if SSB not happened yet, scalar field is 0 everywhere
        if rho_SSB < rho_mean:
            self.u = np.zeros_like(self.rho)
            self.u_inf = 0.
            self.time_taken = 0.
            self.n_iter = 0
            return

        # some useful coeffs
        C = 2 * l0_symm**2 / self.r**2
        C1 = C * self.rout / (self.r * self.hq**2)
        C2 = C * self.rin / (self.r * self.hq**2)
        C3 = C * self.stout2 / self.hct**2
        C4 = C * self.stin2 / self.hct**2

        # calculate p
        t1 = self.rho / rho_SSB
        t2 = C * (self.rout + self.rin) / (self.r * self.hq**2)
        t3 = C * (self.stout2 + self.stin2) / (self.hct**2)
        p = t1 + t2 + t3 - 1

        # start with guess: vev if rho<rhocrit, 0 otherwise
        self.u_inf = np.sqrt(1 - rho_mean / rho_SSB)
        u_guess = np.ones_like(self.rho) * self.u_inf
        if set_zero_region:
            u_guess[self.rho > rho_SSB] = 0
        u = np.copy(u_guess)


        # impose BCs
        u_ext = self._get_extended_field(u)

        # gauss-seidel relaxation
        i = 0
        du_max = 1
        while ((i < imin) or (du_max > tol)):

            # print progress if requested
            if verbose and (i % 1000 == 0):
                print(i, du_max, flush=True)

            # reimpose BCs
            u_ext = self._get_extended_field(u)

            # get offset arrays
            uip = u_ext[2:, 1:-1]
            uim = u_ext[:-2, 1:-1]
            ujp = u_ext[1:-1, 2:]
            ujm = u_ext[1:-1, :-2]

            # red sweep
            q = -C1 * uip - C2 * uim
            q += -C3 * ujp - C4 * ujm
            L = L_op(u, p, q)
            dL = dLdphi_op(u, p)
            du = -(self.red * np.divide(L, dL))
            u += du

            # reimpose BCs
            u_ext = self._get_extended_field(u)

            # get offset arrays
            uip = u_ext[2:, 1:-1]
            uim = u_ext[:-2, 1:-1]
            ujp = u_ext[1:-1, 2:]
            ujm = u_ext[1:-1, :-2]

            # black sweep
            q = -C1 * uip - C2 * uim
            q += -C3 * ujp - C4 * ujm
            L = L_op(u, p, q)
            dL = dLdphi_op(u, p)
            du = -(self.black * np.divide(L, dL))
            u += du

            # calculate residual
            du_max = np.max(np.abs(du))

            i += 1
            if i == imax:
                break

        self.u = u

        # stop timer
        t1 = time()
        self.time_taken = t1 - t0
        self.n_iter = i
        if verbose:
            print(f"Took {self.time_taken:.2f} seconds")
        return

    def _get_extended_field(self, u):
        # u_ext contains BCs: 0 deriv at centre, vev at infinity
        u_ext = np.vstack((u, self.u_inf * np.ones(self.N_th)))  # outer BC
        u_ext = np.vstack((u[0], u_ext))            # inner BC
        #u_ext = np.vstack((0.5*(u[0] + u[1]), u_ext))            # inner BC
        u_ext = np.hstack((u_ext, u_ext[:, -1][:, None]))  # theta=pi/2 BC
        u_ext = np.hstack((u_ext[:, 0][:, None], u_ext))   # theta=0 BC
        return u_ext

    def save(self, filename):

        # check correct file ending
        if filename[-5:] == '.hdf5':
            f = h5py.File(filename, 'w')
        else:
            f = h5py.File(filename + ".hdf5", 'w')

        # set up header group
        header = f.create_group("Header")
        header.attrs['N_radial'] = self.N_r
        header.attrs['N_polar'] = self.N_th
        header.attrs['r_min'] = self.r_min
        header.attrs['r_max'] = self.r_max
        header.attrs['PhiInf'] = self.u_inf
        header.attrs['TimeTaken'] = self.time_taken
        header.attrs['NumIterations'] = self.n_iter

        # save density and scalar field soln.
        f.create_dataset('Density', data=self.rho)
        f.create_dataset('Phi', data=self.u)

        f.close()
        return


def load_solution(filename):

    # check correct file ending
    if filename[-5:] == '.hdf5':
        f = h5py.File(filename, 'r')
    else:
        f = h5py.File(filename + ".hdf5", 'r')

    header = f["Header"]
    N_r = header.attrs['N_radial']
    N_th = header.attrs['N_polar']
    r_min = header.attrs['r_min']
    r_max = header.attrs['r_max']

    solver = Symm2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
    solver.set_density(np.array(f['Density']))
    solver.time_taken = header.attrs['TimeTaken']
    solver.n_iter = header.attrs['NumIterations']
    solver.u_inf = header.attrs['PhiInf']
    solver.u = np.array(f['Phi'])
    return solver
