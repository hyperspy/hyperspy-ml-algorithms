# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

from array_api_compat import array_namespace


class Orthomax:
    """Orthomax rotation of components (varimax when gamma=1.0).

    Computes an orthogonal rotation of the component matrix, preserving
    orthogonality of the components. When ``gamma=1.0`` this is the
    standard varimax rotation, which finds a rotation matrix W that
    maximises the variance of the squared components.

    Parameters
    ----------
    gamma : float, default=1.0
        Orthomax parameter. If ``gamma`` in range ``[0, 1]``, use the
        SVD-based fast algorithm, otherwise solve with a sequence of
        bivariate rotations. The default (1.0) corresponds to varimax.
    tol : float, default=1e-3
        Tolerance of the stopping condition.
    max_iter : int, default=100
        Maximum number of iterations before exiting without convergence.

    Attributes
    ----------
    rotation_matrix_ : ndarray of shape (n_components, n_components)
        The learned rotation matrix.
    components_ : ndarray of shape (n_features, n_components)
        The rotated components matrix (input ``X`` multiplied by
        ``rotation_matrix_``).

    Notes
    -----
    Adapted from metpy.
    """

    def __init__(self, gamma=1.0, tol=1e-3, max_iter=100):
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """Fit the orthomax rotation to the component matrix X.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_components)
            The component matrix to rotate.
        y : ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        self : Orthomax
            Fitted estimator.
        """
        xp = array_namespace(X)
        d, m = X.shape
        oo_d = 1.0 / d

        B = xp.copy(X)
        W = xp.eye(m, dtype=X.dtype)

        if 0.0 <= self.gamma <= 1.0:
            # Use Lawley and Maxwell's fast version
            converged = False
            while not converged:
                S = 0.0
                for _ in range(self.max_iter):  # pragma: no branch
                    Sold = S
                    Bsq = B**2
                    U, S, V = xp.linalg.svd(
                        xp.matrix_transpose(X)
                        @ (d * B * Bsq - self.gamma * B * xp.sum(Bsq, axis=0)),
                        full_matrices=False,
                    )
                    W = U @ V
                    S = xp.sum(S)
                    B = X @ W

                    if abs(S - Sold) < self.tol * S:
                        converged = True
                        break
        else:
            # Use a sequence of bivariate rotations
            for _ in range(self.max_iter):  # pragma: no branch
                maxTheta = 0.0

                for i in range(m - 1):
                    for j in range(i, m):
                        Bi = B[:, i]
                        Bj = B[:, j]
                        u = Bi * Bi - Bj * Bj
                        v = 2.0 * Bi * Bj

                        usum = xp.sum(u)
                        vsum = xp.sum(v)

                        numer = 2.0 * u @ v - 2.0 * self.gamma * usum * vsum * oo_d
                        denom = u @ u - v @ v - self.gamma * (usum**2 - vsum**2) * oo_d

                        theta = 0.25 * xp.arctan2(numer, denom)
                        maxTheta = max(maxTheta, float(xp.abs(theta)))

                        cos_t = xp.cos(theta)
                        sin_t = xp.sin(theta)
                        R = xp.array(
                            [
                                [cos_t, -sin_t],
                                [sin_t, cos_t],
                            ]
                        )

                        B[:, [i, j]] = B[:, [i, j]] @ R
                        W[:, [i, j]] = W[:, [i, j]] @ R

                if maxTheta < self.tol:
                    break

        self.components_ = B
        self.rotation_matrix_ = W
        return self

    def transform(self, X):
        """Apply the learned rotation to a component matrix X.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_components)
            The component matrix to rotate.

        Returns
        -------
        X_rotated : ndarray of shape (n_features, n_components)
            The rotated component matrix.
        """
        return X @ self.rotation_matrix_

    def fit_transform(self, X, y=None):
        """Fit the orthomax rotation to X and return the rotated components.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_components)
            The component matrix to rotate.
        y : ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        components_ : ndarray of shape (n_features, n_components)
            The rotated component matrix.
        """
        return self.fit(X, y).components_
