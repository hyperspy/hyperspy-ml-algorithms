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

"""Incremental (streaming) SVD estimator.

Implements the Ross et al. (2008) incremental SVD algorithm [1]_ for
out-of-core / streaming data.  Data is fed in batches via ``partial_fit``;
after all batches have been processed, use ``transform`` to project new
data onto the learned components.

Unlike PCA, no centering is applied — this is a plain SVD decomposition.

References
----------
.. [1] D. A. Ross, J. Lim, R.-S. Lin, and M.-H. Yang,
   "Incremental Learning for Robust Visual Tracking",
   International Journal of Computer Vision, vol. 77, pp. 125-141, 2008.
"""

import numpy as np
from array_api_compat import array_namespace


def _svd_flip(U, Vt, u_based_decision=False):
    """Flip signs of SVD output for deterministic sign convention.

    Ensures the largest absolute value in each column of U (or row of Vt)
    is positive.  Equivalent to :func:`sklearn.utils.extmath.svd_flip`.
    Works with any array backend detected by ``array_namespace``.

    Parameters
    ----------
    U : array, shape (n_samples, n_components)
        Left singular vectors.
    Vt : array, shape (n_components, n_features)
        Right singular vectors (transposed).
    u_based_decision : bool, default False
        If True, base sign decision on columns of U; otherwise on rows of Vt.

    Returns
    -------
    U : array
        Sign-flipped left singular vectors.
    Vt : array
        Sign-flipped right singular vectors (transposed).
    """
    xp = array_namespace(U, Vt)
    if u_based_decision:
        max_abs_cols = xp.argmax(xp.abs(U), axis=0)
        signs = xp.sign(U[max_abs_cols, xp.arange(U.shape[1])])
    else:
        max_abs_rows = xp.argmax(xp.abs(Vt), axis=1)
        signs = xp.sign(Vt[xp.arange(Vt.shape[0]), max_abs_rows])
    signs = xp.astype(signs, U.dtype)
    U = U * xp.expand_dims(signs, axis=0)
    Vt = Vt * xp.expand_dims(signs, axis=1)
    return U, Vt


class IncrementalSVD:
    """Incremental (streaming) SVD estimator (no centering).

    Computes a plain SVD incrementally by feeding data in batches via
    ``partial_fit``.  After all batches have been processed, the learned
    components and singular values are available as attributes.

    Uses the algorithm of Ross et al. (2008): each new batch is stacked
    with the previous top-*k* subspace (scaled by singular values), then
    a rank-*k* truncated SVD is computed on the stacked matrix to merge
    the new data into the existing subspace.  No mean centering is ever
    applied, so the decomposition is an SVD, not PCA.

    Parameters
    ----------
    n_components : int or None, default None
        Number of singular components to compute.  If None, defaults to
        ``min(n_samples, n_features)`` on the first batch.
    num_chunks : int or None, default None
        Number of chunks to split the data into when calling ``fit()``.
        If None, a heuristic is used based on the data size.  Ignored
        when using ``partial_fit`` directly.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Right singular vectors (rows are components).
    singular_values_ : array, shape (n_components,)
        Singular values in descending order.
    explained_variance_ : array, shape (n_components,)
        Variance explained by each component (``S² / N``).
    explained_variance_ratio_ : array, shape (n_components,)
        Fraction of top-*k* variance captured by each component
        (``S² / sum(S²)``).
    mean_ : array, shape (n_features,)
        Always zeros — no centering is applied.  Provided for API
        compatibility with estimators that do centre.
    n_samples_seen_ : int
        Total number of samples processed across all ``partial_fit`` calls.
    noise_variance_ : float
        Mean of discarded singular values squared, divided by
        ``n_samples_seen_`` (if any singular values were discarded).

    Examples
    --------
    >>> import numpy as np
    >>> from hyperspy_ml_algorithms import IncrementalSVD
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((200, 50))
    >>> est = IncrementalSVD(n_components=3)
    >>> for chunk in np.array_split(X, 4):
    ...     est.partial_fit(chunk)
    >>> components = est.components_.T       # shape (n_features, n_components)
    >>> scores = est.transform(X)            # shape (n_samples, n_components)
    """

    def __init__(self, n_components=None, num_chunks=None):
        self.n_components = n_components
        self.num_chunks = num_chunks
        self.components_ = None
        self.singular_values_ = None
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        """Fit the incremental SVD model to X.

        Splits the data into chunks and calls ``partial_fit`` on each.
        Supports NumPy, CuPy, PyTorch, and Dask arrays.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.  Can be a NumPy, CuPy, PyTorch, or Dask array.
        y : Ignored
            Exists for API compatibility.

        Returns
        -------
        self : IncrementalSVD
            The fitted estimator.
        """
        # Reset state for a fresh fit.
        self.components_ = None
        self.singular_values_ = None
        self.n_samples_seen_ = 0
        if hasattr(self, "_mean"):
            del self._mean

        n_samples = X.shape[0]

        if self.num_chunks is None:
            n_chunks = max(1, min(max(4, n_samples // 100), 50))
        else:
            n_chunks = self.num_chunks

        # Detect dask arrays and iterate blocks, materialising each chunk
        # as a NumPy array.
        if hasattr(X, "blocks"):
            for i in range(n_chunks):
                start = i * n_samples // n_chunks
                end = (i + 1) * n_samples // n_chunks if i < n_chunks - 1 else n_samples
                chunk = np.asarray(X[start:end])
                self.partial_fit(chunk)
        else:
            for i in range(n_chunks):
                start = i * n_samples // n_chunks
                end = (i + 1) * n_samples // n_chunks if i < n_chunks - 1 else n_samples
                self.partial_fit(X[start:end])

        return self

    def partial_fit(self, X_chunk, y=None):
        """Fit one batch without centering (plain incremental SVD).

        Implements the Ross et al. (2008) rank-1 update algorithm.
        Data is never mean-subtracted, so the decomposition is a plain
        SVD rather than PCA.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
            Batch of training data.  Can be a NumPy, CuPy, PyTorch, or
            materialised Dask array.
        y : Ignored
            Exists for API compatibility.

        Returns
        -------
        self : IncrementalSVD
            The fitted estimator.
        """
        xp = array_namespace(X_chunk)

        n_samples, n_features = X_chunk.shape

        # --- first-call initialization ---
        if self.components_ is None:
            self.n_samples_seen_ = 0

            if self.n_components is None:
                self.n_components_ = min(n_samples, n_features)
            elif self.n_components > n_features:
                raise ValueError(
                    f"n_components={self.n_components} must be less "
                    f"than or equal to n_features={n_features}"
                )
            elif self.n_components > n_samples:
                raise ValueError(
                    f"n_components={self.n_components} must be less "
                    f"than or equal to the batch number of samples "
                    f"{n_samples} for the first partial_fit call."
                )
            else:
                self.n_components_ = self.n_components

        elif self.components_.shape[0] != self.n_components_:
            raise ValueError(
                f"n_components has changed from "
                f"{self.components_.shape[0]} to "
                f"{self.n_components_} between calls to "
                f"partial_fit!"
            )

        # --- build the matrix for SVD ---
        if self.n_samples_seen_ == 0:
            # First batch: plain SVD of raw (uncentered) data.
            X_stacked = X_chunk
        else:
            # Subsequent batch: stack previous top-k subspace (scaled
            # by singular values) with the new raw batch — the Ross
            # et al. (2008) incremental SVD algorithm, without
            # centering or mean-correction.
            prev = self.singular_values_.reshape((-1, 1)) * self.components_
            X_stacked = xp.concat([prev, X_chunk], axis=0)

        # SVD through the array namespace stays on the input device (NumPy,
        # CuPy, or PyTorch).
        U, S, Vt = xp.linalg.svd(X_stacked, full_matrices=False)
        U, Vt = _svd_flip(U, Vt, u_based_decision=False)

        self.n_samples_seen_ += n_samples
        k = self.n_components_
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]

        # --- explained_variance_ (S² / N) ---
        n_total = self.n_samples_seen_
        self.explained_variance_ = (S[:k] ** 2) / n_total

        # --- explained_variance_ratio_ (S² / sum(S²) for top-k) ---
        top_k_ev = S[:k] ** 2
        self.explained_variance_ratio_ = top_k_ev / top_k_ev.sum()

        if k not in (n_samples, n_features):
            self.noise_variance_ = (S[k:] ** 2).mean() / n_total
        else:
            self.noise_variance_ = 0.0

        self._mean = xp.zeros(n_features)

        return self

    def transform(self, X):
        """Project X onto the learned components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to project.  Can be a NumPy, CuPy, PyTorch, or Dask array.

        Returns
        -------
        scores : array, shape (n_samples, n_components)
            Projected data (scores).  Type matches the input backend.
        """
        # X @ components_.T works for NumPy, CuPy, PyTorch, and Dask arrays
        # without needing array_namespace dispatch.
        return X @ self.components_.T

    def fit_transform(self, X, y=None):
        """Fit the model and transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : Ignored
            Exists for API compatibility.

        Returns
        -------
        scores : ndarray, shape (n_samples, n_components)
            Projected data (scores).
        """
        self.fit(X)
        return self.transform(X)

    @property
    def mean_(self):
        """Mean vector (always zeros — no centering is applied).

        Returns a scalar ``0.0`` before any ``partial_fit`` call, and
        a zeros array of shape ``(n_features,)`` afterward.
        """
        return self.__dict__.get("_mean", 0.0)

    @mean_.setter
    def mean_(self, value):
        # Allow ``mean_ = 0.0`` (sklearn init convention) and
        # ``mean_ = array(...)`` (post-fit assignment).  Always store
        # zeros so that centering has no effect, preserving the input
        # array backend.
        if np.asarray(value).ndim == 0:
            self.__dict__["_mean"] = float(value)
        else:
            xp = array_namespace(value)
            self.__dict__["_mean"] = xp.zeros_like(xp.asarray(value))
