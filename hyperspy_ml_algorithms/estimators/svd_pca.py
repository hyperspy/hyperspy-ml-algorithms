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

import importlib
import logging

import numpy as np
from array_api_compat import array_namespace

SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None

_logger = logging.getLogger(__name__)


def svd_flip_signs(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the components in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, v : ndarray
        u and v are the outputs of a singular value decomposition.
    u_based_decision : bool, default True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u, v : ndarray
        Adjusted outputs with same dimensions as inputs.

    """
    # Derived from `sklearn.utils.extmath.svd_flip`.
    # Copyright (c) 2007-2020 The scikit-learn developers.
    # All rights reserved.

    xp = array_namespace(u, v)

    if u_based_decision:
        max_abs_cols = xp.argmax(xp.abs(u), axis=0)
        signs = xp.sign(u[max_abs_cols, xp.arange(u.shape[1])])
    else:
        max_abs_rows = xp.argmax(xp.abs(v), axis=1)
        signs = xp.sign(v[xp.arange(v.shape[0]), max_abs_rows])

    u = u * signs
    v = v * signs[:, None]

    return u, v


def svd_solve(
    data,
    output_dimension=None,
    svd_solver="auto",
    svd_flip=True,
    u_based_decision=True,
    **kwargs,
):
    """Apply singular value decomposition to input data.

    Parameters
    ----------
    data : ndarray
        Input data array with shape (m, n)
    output_dimension : None or int
        Number of components to keep/calculate
    svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
        - If ``"auto"``:
          The solver is selected by a default policy based on `data.shape` and
          `output_dimension`: if the input data is larger than 500x500 and the
          number of components to extract is lower than 80% of the smallest
          dimension of the data, then the more efficient "randomized"
          method is enabled. Otherwise the exact full SVD is computed and
          optionally truncated afterwards.
        - If ``"full"``:
          Run exact SVD, calling the standard LAPACK solver via
          :func:`scipy.linalg.svd`, and select the components by postprocessing
        - If ``"arpack"``:
          Use truncated SVD, calling ARPACK solver via
          :func:`scipy.sparse.linalg.svds`. It requires strictly
          `0 < output_dimension < min(data.shape)`
        - If ``"randomized"``:
          Use truncated SVD, calling :func:`sklearn.utils.extmath.randomized_svd`
          to estimate a limited number of components
    svd_flip : bool, default True
        If True, adjusts the signs of the components and scores such that
        the components that are largest in absolute value are always positive.
        See :func:`svd_flip_signs` for more details.
    u_based_decision : bool, default True
        If True, and svd_flip is True, use the columns of u as the basis for sign-flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    U, S, V : ndarray
        Output of SVD such that X = U*S*V.T

    """
    # Derived from `sklearn.decomposition.PCA`.
    # Copyright (c) 2007-2020 The scikit-learn developers.
    # All rights reserved.

    xp = array_namespace(data)

    m, n = data.shape

    if output_dimension is None:
        output_dimension = min(m, n)
        if svd_solver == "arpack":
            output_dimension -= 1

    if svd_solver == "auto":
        if max(m, n) <= 500:
            svd_solver = "full"
        elif (
            output_dimension >= 1
            and output_dimension < 0.8 * min(m, n)
            and SKLEARN_INSTALLED
        ):
            svd_solver = "randomized"
        else:
            svd_solver = "full"

    if svd_solver == "randomized":
        if not SKLEARN_INSTALLED:
            raise ImportError(
                "svd_solver='randomized' requires scikit-learn to be installed"
            )
        import sklearn

        # sklearn.randomized_svd requires numpy arrays
        _data = np.asarray(data) if hasattr(data, "get") else data
        U, S, V = sklearn.utils.extmath.randomized_svd(
            _data, n_components=output_dimension, **kwargs
        )
    elif svd_solver == "arpack":
        if output_dimension >= min(m, n):
            raise ValueError(
                "svd_solver='arpack' requires output_dimension "
                "to be strictly less than min(data.shape)."
            )
        # Dispatch based on array namespace for GPU support
        if xp.__name__ == "cupy":  # pragma: no cover
            from cupyx.scipy.sparse.linalg import svds
        else:
            from scipy.sparse.linalg import svds
        U, S, V = svds(data, k=output_dimension, **kwargs)
        # svds doesn't follow scipy.linalg.svd conventions,
        # so reverse its outputs
        S = S[::-1]
        # flip eigenvectors' sign to enforce deterministic output
        if svd_flip:
            U, V = svd_flip_signs(
                U[:, ::-1], V[::-1], u_based_decision=u_based_decision
            )
    elif svd_solver == "full":
        U, S, Vt = xp.linalg.svd(data, full_matrices=False)
        V = Vt
        # flip eigenvectors' sign to enforce deterministic output
        if svd_flip:
            U, V = svd_flip_signs(U, V, u_based_decision=u_based_decision)

        U = U[:, :output_dimension]
        S = S[:output_dimension]
        V = V[:output_dimension, :]

    return U, S, V


class SVDPCA:
    """SVD-based PCA estimator.

    Performs Principal Component Analysis using Singular Value Decomposition.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components to keep. If None, keep all components.
    svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
        SVD solver to use. See :func:`svd_solve` for details.
    centre : {None, "signal", "features", False}, default None
        Centering strategy:

        - ``None`` or ``False``: no centering.
        - ``"signal"``: center along the signal axis (each sample's
          signal values are centered around zero).
        - ``"features"``: center the features (each feature is centered
          to have zero mean across samples). This is the standard PCA
          centering (equivalent to sklearn's PCA).
    auto_transpose : bool, default True
        If True, automatically transposes the data to boost performance.
    svd_flip : bool, default True
        If True, adjusts the signs of the components and scores such that
        the components that are largest in absolute value are always positive.
    u_based_decision : bool, default True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v.

    Attributes
    ----------
    components_ : ndarray of shape (n_features, n_components)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        decreasing singular values.
    singular_values_ : ndarray of shape (n_components,)
        Singular values corresponding to each component.
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each component.
        When ``centre`` is not None (mean-centered data), this is the
        variance explained by each component (``\u03c3\u1d62\u00b2 / N``), consistent
        with PCA. When ``centre`` is None (no centering), this is the
        mean squared contribution of each component (``\u03c3\u1d62\u00b2 / N``),
        i.e. a measure of signal energy rather than variance.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component.
    mean_ : ndarray or None
        Per-feature empirical mean, estimated from the training data.
        None if ``centre`` is None or False.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspy_ml_algorithms import SVDPCA
    >>> rng = np.random.RandomState(42)
    >>> X = rng.random((77, 13))
    >>> est = SVDPCA(n_components=5)
    >>> est.fit(X)
    SVDPCA(n_components=5, ...)
    >>> est.components_.shape
    (5, 13)
    >>> scores = est.transform(X)
    >>> scores.shape
    (77, 5)
    """

    def __init__(
        self,
        n_components=None,
        svd_solver="auto",
        centre=None,
        auto_transpose=True,
        svd_flip=True,
        u_based_decision=True,
    ):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.centre = centre
        self.auto_transpose = auto_transpose
        self.svd_flip = svd_flip
        self.u_based_decision = u_based_decision

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        xp = array_namespace(X)
        data = xp.asarray(X, dtype=xp.float64)
        N, M = data.shape

        centre = self.centre
        if centre is None or centre is False:
            mean = None
        else:
            if centre == "signal":
                mean = xp.mean(data, axis=1)[:, None]
            elif centre == "features":
                mean = xp.mean(data, axis=0)[None, :]
            else:
                raise ValueError(
                    "'centre' must be one of [None, False, 'signal', 'features']"
                )
            data = data - mean

        auto_transpose = self.auto_transpose
        if auto_transpose:
            if N < M:
                _logger.info("Auto-transposing the data")
                # Transpose: swap axes 0 and 1
                data = xp.permute_dims(data, (1, 0))
            else:
                auto_transpose = False

        U, S, V = svd_solve(
            data,
            output_dimension=self.n_components,
            svd_solver=self.svd_solver,
            svd_flip=self.svd_flip,
            u_based_decision=self.u_based_decision,
        )

        # \u03c3\u1d62\u00b2/N: equals explained variance when data is mean-centered (PCA),
        # or mean squared signal contribution (proportion of total variation)
        # when data is not centered.
        explained_variance = S**2 / N

        if not auto_transpose:
            components = V
            scores = U * S
        else:
            scores = V.T
            # Transpose back: U has shape (n_features, n_components),
            # we want (n_components, n_features) for sklearn convention.
            components = xp.permute_dims(U * S, (1, 0))

        # Store fitted attributes
        self.components_ = components
        self.singular_values_ = S
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / xp.sum(explained_variance)
        self.mean_ = mean

        # Store scores from fit for fit_transform
        self._scores = scores

        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        Project X onto the learned components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data (scores).
        """
        xp = array_namespace(X)
        X = xp.asarray(X, dtype=xp.float64)

        if self.mean_ is not None:
            X = X - self.mean_

        return xp.matmul(X, self.components_.T)

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data (scores).
        """
        self.fit(X)
        return self._scores

    def __repr__(self):
        params = [
            f"n_components={self.n_components!r}",
            f"svd_solver={self.svd_solver!r}",
            f"centre={self.centre!r}",
            f"auto_transpose={self.auto_transpose!r}",
        ]
        return f"SVDPCA({', '.join(params)})"
