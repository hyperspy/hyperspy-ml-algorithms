# -*- coding: utf-8 -*-
# This file is a transcription of a MATLAB code obtained from the
# following research paper: Darren T. Andrews and Peter D. Wentzell,
# "Applications of maximum likelihood principal component analysis:
# incomplete data sets and calibration transfer,"
# Analytica Chimica Acta 350, no. 3 (September 19, 1997): 341-352.
#
# Copyright 1997 Darren T. Andrews and Peter D. Wentzell
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

import logging

import numpy as np
import scipy.linalg
from array_api_compat import array_namespace

_logger = logging.getLogger(__name__)


def _svd(x, xp):
    """Compute full SVD via scipy, converting between array namespaces.

    Parameters
    ----------
    x : array
        Input matrix of shape (m, n).
    xp : module
        Array namespace module (e.g. numpy, cupy).

    Returns
    -------
    U, S, V : arrays in the *xp* namespace
        SVD decomposition such that x = U @ diag(S) @ V.T.
    """
    x_np = np.asarray(x)
    U_np, S_np, V_np = scipy.linalg.svd(x_np, full_matrices=False)
    return xp.asarray(U_np), xp.asarray(S_np), xp.asarray(V_np)


class MLPCA:
    """Maximum Likelihood Principal Component Analysis.

    Standard PCA based on a singular value decomposition (SVD) approach assumes
    that the data is corrupted with Gaussian, or homoskedastic noise. For many
    applications, this assumption does not hold. For example, count data from
    EDS-TEM experiments is corrupted by Poisson noise, where the noise variance
    depends on the underlying pixel value. Rather than scaling or transforming
    the data to approximately "normalize" the noise, MLPCA instead uses estimates
    of the data variance to perform the decomposition.

    This implementation is a transcription of MATLAB code from [Andrews1997]_.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. If ``None``, all components are kept.
    max_iter : int, default=50000
        Maximum number of iterations before exiting without convergence.
    tol : float, default=1e-10
        Relative tolerance of the stopping condition.  When the objective
        function values are very small (near machine epsilon), the
        `abs_tol` threshold takes precedence.
    abs_tol : float, default=1e-12
        Absolute tolerance.  If the objective function ``s_obj`` drops
        below this value, convergence is declared immediately regardless
        of the relative criterion.  This prevents the algorithm from
        running unnecessary iterations when the residual is already
        negligible.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance. Equivalent to the right singular vectors (V).
    singular_values_ : ndarray of shape (n_components,)
        Singular values corresponding to each component.
    scores_ : ndarray of shape (n_samples, n_components)
        Projection of the data onto the components (U * S).
    mean_ : None
        MLPCA does not center the data; always ``None``.

    References
    ----------
    .. [Andrews1997] Darren T. Andrews and Peter D. Wentzell, "Applications
        of maximum likelihood principal component analysis: incomplete
        data sets and calibration transfer", Analytica Chimica Acta 350,
        no. 3 (September 19, 1997): 341-352.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspy_ml_algorithms import MLPCA
    >>> rng = np.random.RandomState(42)
    >>> X = rng.poisson(10, size=(50, 30)).astype(float)
    >>> variance = X.copy()  # Poisson noise: variance = mean
    >>> est = MLPCA(n_components=3)
    >>> est.fit(X, variance)
    >>> scores = est.transform(X)
    >>> scores.shape
    (50, 3)
    """

    def __init__(self, n_components=None, max_iter=50000, tol=1e-10, abs_tol=1e-12):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.abs_tol = abs_tol

    def fit(self, X, variance):
        """Fit the MLPCA model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        variance : array-like of shape (n_samples, n_features)
            Per-element variance estimates for X. Zeros indicate missing
            measurements. For Poisson-distributed data, ``variance = X``.

        Returns
        -------
        self : MLPCA
            Fitted estimator.
        """
        xp = array_namespace(X, variance)

        m, n = X.shape
        output_dimension = self.n_components
        if output_dimension is None:
            output_dimension = min(m, n)

        # Handle zero/infinite variance (missing data)
        inv_v = xp.where(xp.isfinite(1.0 / variance), 1.0 / variance, 1.0)

        _logger.info("Performing maximum likelihood principal components analysis")

        # Generate initial estimates via SVD of data directly
        # (equivalent to eigenvectors of X @ X.T but avoids forming the
        # (m, m) covariance matrix when m >> n)
        _logger.info("Generating initial estimates")
        X_centered = X - xp.mean(X, axis=1, keepdims=True)
        U, _, _ = _svd(X_centered, xp)
        U = U[:, :output_dimension]

        s_old = 0.0

        # Placeholders
        M = xp.zeros((m, n))

        # Loop for alternating least squares
        _logger.info("Optimization iteration loop")
        for itr in range(self.max_iter):  # pragma: no branch
            s_obj = 0.0

            for i in range(n):
                Uq = U.T * inv_v[:, i]
                # Solve (Uq @ U) z = Uq @ x  instead of inv(Uq @ U) @ Uq @ x
                M[:, i] = U @ xp.linalg.solve(Uq @ U, Uq @ X[:, i])
                dx = X[:, i] - M[:, i]
                s_obj += (dx * inv_v[:, i]) @ dx.T

            # Every second iteration, check the stop criterion
            if itr > 0 and itr % 2 == 0:
                # Absolute tolerance: if objective is near-zero, we have converged
                if s_obj < self.abs_tol:
                    _logger.info(f"Iteration: {itr // 2}, s_obj={s_obj:.2e} < abs_tol, converged")
                    break
                stop_criterion = abs(s_old - s_obj) / s_obj
                _logger.info(f"Iteration: {itr // 2}, convergence: {stop_criterion}")

                if stop_criterion < self.tol:
                    break

            # Transpose for next iteration
            s_old = s_obj
            _, _, V = _svd(M, xp)

            X = X.T
            inv_v = inv_v.T
            M = M.T

            m, n = X.shape
            U = V[:output_dimension].T

        # Final SVD to obtain components and scores
        U, S, Vt = _svd(M, xp)

        self.components_ = Vt[:output_dimension]
        self.singular_values_ = S[:output_dimension]
        self.mean_ = None
        self.scores_ = U[:, :output_dimension] * S[:output_dimension]

        return self

    def transform(self, X):
        """Project X onto the learned components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            Projection of X onto the components.
        """
        _ = array_namespace(X, self.components_)  # noqa: F841 — namespace dispatch
        return X @ self.components_.T

    def fit_transform(self, X, variance):
        """Fit the model and return the scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        variance : array-like of shape (n_samples, n_features)
            Per-element variance estimates for X.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            Projection of X onto the components.
        """
        self.fit(X, variance)
        return self.scores_
