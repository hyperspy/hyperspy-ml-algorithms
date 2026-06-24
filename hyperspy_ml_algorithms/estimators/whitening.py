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

from hyperspy_ml_algorithms.utils._array_namespace import array_namespace


class Whitening:
    """Decorrelate variables via a whitening transformation.

    A whitening transformation decorrelates the variables such that the
    covariance matrix of the whitened data is the identity matrix.

    If *X* is a random vector with non-singular covariance matrix *C*,
    and *W* is a whitening matrix satisfying *W^T W = C^{-1}*, then the
    transformation *Y = X @ W^T* yields a whitened random vector *Y*
    with unit diagonal covariance.  In ZCA whitening the matrix
    *W = C^{-1/2}*, while in PCA whitening *W* is derived from the
    eigensystem of *C*.  More details can be found in [Kessy2015]_.

    Parameters
    ----------
    method : {'PCA', 'ZCA'}, default='PCA'
        Whitening method.  PCA whitening is the default for
        backward compatibility with HyperSpy < 1.6.0.
    centre : bool, default=True
        If True, centre the data by subtracting the feature-wise mean
        before computing the whitening transform.
    epsilon : float, default=1e-10
        Small constant added to eigenvalues before taking the inverse
        square root to avoid division by zero.

    Attributes
    ----------
    whitening_matrix_ : array-like, shape (n_features, n_features)
        Learned whitening matrix *W* such that
        ``Y = (X - self.mean_) @ self.whitening_matrix_.T``.
    mean_ : array-like, shape (n_features,)
        Feature-wise mean of the training data.  Set to zeros when
        ``centre=False``.

    Notes
    -----
    The ``y`` parameter in ``fit`` and ``fit_transform`` is ignored
    (sklearn API compatibility).

    References
    ----------
    .. [Kessy2015] A. Kessy, A. Lewin, and K. Strimmer, "Optimal
        Whitening and Decorrelation", arXiv:1512.00809, (2015),
        https://arxiv.org/pdf/1512.00809.pdf
    """

    def __init__(self, method="PCA", centre=True, epsilon=1e-10):
        self.method = method
        self.centre = centre
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Compute the whitening matrix from the data *X*.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self : Whitening
            Fitted estimator.
        """
        xp = array_namespace(X)

        if self.centre:
            self.mean_ = xp.mean(X, axis=0)
            Y = X - self.mean_
        else:
            self.mean_ = xp.zeros((X.shape[1],), dtype=X.dtype)
            Y = X

        # Covariance matrix
        R = (Y.T @ Y) / Y.shape[0]

        # SVD of the covariance matrix
        U, S, _ = xp.linalg.svd(R, full_matrices=False)
        # Reshape S from (n,) to (n, 1) for broadcasting
        S_safe = xp.sqrt(S + self.epsilon)
        S_safe = xp.reshape(S_safe, (S_safe.shape[0], 1))

        if self.method == "PCA":
            self.whitening_matrix_ = U.T / S_safe
        elif self.method == "ZCA":
            self.whitening_matrix_ = U @ (U.T / S_safe)
        else:
            raise ValueError(f"method must be one of ['PCA', 'ZCA'], got {self.method}")

        return self

    def transform(self, X):
        """Whiten data *X* using the learned whitening matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to whiten.

        Returns
        -------
        Y : array-like, shape (n_samples, n_features)
            Whitened data.
        """

        if self.centre:
            Y = X - self.mean_
        else:
            Y = X

        return Y @ self.whitening_matrix_.T

    def fit_transform(self, X, y=None):
        """Fit to *X* then whiten it in one call.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        Y : array-like, shape (n_samples, n_features)
            Whitened data.
        """
        return self.fit(X, y).transform(X)
