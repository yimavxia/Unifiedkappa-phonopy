# Copyright (C) 2013 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import warnings
import textwrap
import numpy as np
from phonopy.units import VaspToTHz
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.degeneracy import degenerate_sets


def get_group_velocity(q,  
                       dynamical_matrix,
                       q_length=None,  
                       symmetry=None,
                       frequency_factor_to_THz=VaspToTHz):
    """Returns group velocity at a q-point."""
    gv = GroupVelocity(dynamical_matrix,
                       q_length=q_length,
                       symmetry=symmetry,
                       frequency_factor_to_THz=frequency_factor_to_THz)
    gv.run([q])
    return gv.group_velocity[0]


def delta_dynamical_matrix(q,
                           delta_q,
                           dynmat):
    dynmat.run(q - delta_q)
    dm1 = dynmat.dynamical_matrix
    dynmat.run(q + delta_q)
    dm2 = dynmat.dynamical_matrix
    return dm2 - dm1


class GroupVelocity(object):
    
    """Class to calculate group velocities of phonons

    d omega   ----
    ------- = \  / omega
    d q        \/q

    Gradient of omega in reciprocal space, which is calculated here by

       1             d D(q)
    ------- <e(q,nu)|------|e(q,nu)>
    2 omega           d q

    Attributes
    ----------
    group_velocity : ndarray
        Group velocities at q-points.
        shape=(q-points, 3), dtype='double', order='C'
    q_length : float
        Distance in reciprocal space used to calculate finite difference of
        dynamcial matrix.

    """

    def __init__(self,
                 dynamical_matrix,
                 q_length=None,
                 symmetry=None,
                 frequency_factor_to_THz=VaspToTHz,
                 cutoff_frequency=1e-4):
        """

        dynamical_matrix : DynamicalMatrix or DynamicalMatrixNAC
            Dynamical matrix class instance.
        q_length : float
            This is used such as D(q + q_length) - D(q - q_length) for
            calculating finite difference of dynamical matrix.
            Default is None, which gives 1e-5.
        symmetry : Symmetry
            This is used to symmetrize group velocity at each q-points.
            Default is None, which means no symmetrization.
        frequency_factor_to_THz : float
            Unit conversion factor to convert to THz. Default is VaspToTHz.
        cutoff_frequency : float
            Group velocity is set zero if phonon frequency is below this value.

        """
        self._dynmat = dynamical_matrix
        primitive = dynamical_matrix.get_primitive()
        self._reciprocal_lattice_inv = primitive.get_cell()
        self._reciprocal_lattice = np.linalg.inv(self._reciprocal_lattice_inv)
        self._q_length = q_length
        if self._dynmat.is_nac() and self._dynmat.get_nac_method() == 'gonze':
            if self._q_length is None:
                self._q_length = 1e-5
        if self._q_length is None:
            self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        else:
            self._ddm = None
        #set None None type
        self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        self._symmetry = symmetry
        self._factor = frequency_factor_to_THz
        self._cutoff_frequency = cutoff_frequency

        self._directions = np.array([[1, 2, 3],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]], dtype='double')
        self._directions[0] /= np.linalg.norm(self._directions[0])

        self._q_points = None
        self._group_velocities = None
        self._perturbation = None

    def run(self, q_points, perturbation=None):
        """Group velocities are computed at q-points.

        q_points : Array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : Array-like
            Direction in fractional coordinates of reciprocal space.

        """

        self._q_points = q_points
        self._perturbation = perturbation
        if perturbation is None:
            self._directions[0] = np.array([1, 2, 3])
        else:
            self._directions[0] = np.dot(
                self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])
        if False:
            gv = [self._calculate_group_velocity_at_q(q) for q in self._q_points]
        else:
            gv = [self._calculate_group_velocity_at_q_xy(q) for q in self._q_points]
        self._group_velocities = np.array(gv, dtype=np.complex128, order='C')

    @property
    def q_length(self):
        return self._q_length

    def get_q_length(self):
        return self.q_length

    @q_length.setter
    def q_length(self, q_length):
        self._q_length = q_length

    def set_q_length(self, q_length):
        self.q_length = q_length

    @property
    def group_velocities(self):
        return self._group_velocities

    def get_group_velocity(self):
        return self.group_velocities

    def _calculate_group_velocity_at_q_xy(self, q):
        '''
        @yixia
        added and modified method to compute off-diagonal group velocities
        '''
        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        deg_sets = degenerate_sets(freqs)
        gv_full = np.zeros((len(freqs),len(freqs),3), dtype=np.complex128, order='C')
        gv = np.zeros((len(freqs),3), dtype=np.complex128, order='C')
        ddms = self._get_dD(np.array(q))
        fdim=len(freqs)
        import copy
        eigvecs_new = copy.deepcopy(eigvecs)
        for deg in deg_sets:
            eigsets = eigvecs[:, deg]
            _, eigvecs_tmp = np.linalg.eigh(np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))
            rot_eigsets = np.dot(eigsets, eigvecs_tmp)
            eigvecs_new [:, deg] = rot_eigsets             
        for i in range(fdim):
            for j in range(fdim):
                for k in range(3):
                    gv_full[i,j,k] = np.dot(eigvecs_new[:,i].T.conj(), np.dot(ddms[k+1],eigvecs_new[:,j]))
                if (freqs[i] > self._cutoff_frequency) and (freqs[j] > self._cutoff_frequency):
                    gv_full[i,j,:] *= self._factor**2/(freqs[i]+freqs[j]) 
                else:
                    gv_full[i,j,:] = 0                    
        for i in range(fdim):
            gv[i,:]=gv_full[i,i,:]
        # return
        return gv_full


    def _calculate_group_velocity_at_q(self, q):
        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        gv = np.zeros((len(freqs), 3), dtype='double', order='C')
        deg_sets = degenerate_sets(freqs)

        ddms = self._get_dD(np.array(q))
        pos = 0       
        for deg in deg_sets:
            if False:
                gv[pos:pos+len(deg)] = self._perturb_D(ddms, eigvecs[:, deg])
            else:
                gv[pos:pos+len(deg)] = np.transpose([
                    np.diag(np.dot(eigvecs[:,deg].T.conj(),np.dot(ddms[1],eigvecs[:, deg]))).real,
                    np.diag(np.dot(eigvecs[:,deg].T.conj(),np.dot(ddms[2],eigvecs[:, deg]))).real,
                    np.diag(np.dot(eigvecs[:,deg].T.conj(),np.dot(ddms[3],eigvecs[:, deg]))).real])
            pos += len(deg)

        for i, f in enumerate(freqs):
            if f > self._cutoff_frequency:
                gv[i, :] *= self._factor ** 2 / f / 2
            else:
                gv[i, :] = 0

        if self._perturbation is None:
            if self._symmetry is None:
                return gv
            else:
                return self._symmetrize_group_velocity(gv, q)
        else:
            return gv

    def _symmetrize_group_velocity(self, gv, q):
        """Symmetrize obtained group velocities using site symmetries."""

        rotations = []
        for r in self._symmetry.get_reciprocal_operations():
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - np.dot(r, q_in_BZ)
            if (np.abs(diff) < self._symmetry.get_symmetry_tolerance()).all():
                rotations.append(r)
                
        gv_sym = np.zeros_like(gv)
        for r in rotations:
            r_cart = similarity_transformation(self._reciprocal_lattice, r)
            gv_sym += np.dot(r_cart, gv.T).T

        return gv_sym / len(rotations)

    def _get_dD(self, q):
        """Compute derivative or finite difference of dynamcial matrices"""

        self._q_length = None
        
        if self._q_length is None:
            return self._get_dD_analytical(q)
        else:
            return self._get_dD_FD(q)

    def _get_dD_FD(self, q):
        """Compute finite difference of dynamcial matrices"""

        ddm = []
        for dqc in self._directions * self._q_length:
            print ("dqc------------------")
            print (dqc)
            dq = np.dot(self._reciprocal_lattice_inv, dqc)
            ddm.append(delta_dynamical_matrix(q, dq, self._dynmat) /
                       self._q_length / 2)
        print ("Shape of ddm --> finite difference")
        print (self._directions * self._q_length)
        print ((np.array(ddm)).shape)
        return np.array(ddm)

    def _get_dD_analytical(self, q):
        """Compute derivative of dynamcial matrices"""

        self._ddm.run(q)
        ddm = self._ddm.get_derivative_of_dynamical_matrix()
        dtype = "c%d" % (np.dtype('double').itemsize * 2)
        ddm_dirs = np.zeros((len(self._directions),) + ddm.shape[1:],
                            dtype=dtype)
        for i, dq in enumerate(self._directions):
            for j in range(3):
                ddm_dirs[i] += dq[j] * ddm[j]
        return ddm_dirs

    def _perturb_D(self, ddms, eigsets):
        """Treat degeneracy

        Group velocities are calculated using analytical continuation using
        specified directions (self._directions) in reciprocal space.

        ddms : Array-like
            List of delta (derivative or finite difference) of dynamical
            matrices along several q-directions for perturbation.
        eigsets : Array-like
            List of phonon eigenvectors of degenerate bands.

        """

        eigvals, eigvecs = np.linalg.eigh(
            np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))

        gv = []
        rot_eigsets = np.dot(eigsets, eigvecs)
        for ddm in ddms[1:]:
            gv.append(
                np.diag(np.dot(rot_eigsets.T.conj(),
                               np.dot(ddm, rot_eigsets))).real)
        return np.transpose(gv)
