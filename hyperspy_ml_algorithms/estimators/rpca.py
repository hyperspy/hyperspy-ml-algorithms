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

"""Robust PCA estimators.

Provides two estimator classes for Robust Principal Component Analysis
(RPCA), which decomposes a data matrix :math:`X` into low-rank :math:`L`
and sparse :math:`S` components such that :math:`X \\approx L + S`.

- ``ORPCA`` : Online Robust PCA for streaming / incremental decomposition.
- ``RPCAGoDec`` : Batch Robust PCA via the GoDec algorithm.
"""

import logging
from itertools import chain

import numpy as np
import scipy
from array_api_compat import array_namespace

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # pragma: no cover
        return iterable


_logger = logging.getLogger(__name__)

__all__ = [
    "ORPCA",
    "RPCAGoDec",
]


def _check_random_state(seed):
    """Turn *seed* into a :class:`numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int, or numpy.random.RandomState
        If None, return a new RandomState instance.
        If int, return a new RandomState instance seeded with *seed*.
        If RandomState, return it unchanged.

    Returns
    -------
    numpy.random.RandomState
    """
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a numpy.random.RandomState instance"
    )


def _soft_thresh(X, lambda1):
    """Soft-thresholding of array *X*.

    Parameters
    ----------
    X : ndarray
        Input array.
    lambda1 : float
        Threshold value.

    Returns
    -------
    ndarray
        Soft-thresholded array ``sign(X) * max(|X| - lambda1, 0)``.
    """
    xp = array_namespace(X)
    res = xp.abs(X) - lambda1
    res = xp.maximum(res, xp.asarray(0.0, device=getattr(X, "device", None)))
    return res * xp.sign(X)


def _solveproj(z, X, Id, lambda2, r=None, e=None):
    """Solve the projection subproblem for ORPCA.

    Given a data vector *z* and current subspace basis *X*, solve for the
    scores ``r`` and sparse error ``e`` that minimise
    ``0.5 * ||z - X @ r - e||² + lambda2 * ||e||₁``.

    Uses block-coordinate descent alternating between *r* and *e* updates.

    Parameters
    ----------
    z : ndarray, shape (m,) or (m, batch_size)
        Data vector(s).
    X : ndarray, shape (m, n)
        Subspace basis (components in columns).
    Id : ndarray, shape (n, n)
        Regularisation matrix (``lambda1 * I``).
    lambda2 : float
        Sparse error regularisation parameter.
    r : ndarray or None, shape (n,) or (n, batch_size)
        Initial guess for the scores. If None, zeros are used.
    e : ndarray or None, shape (m,) or (m, batch_size)
        Initial guess for the sparse error. If None, zeros are used.

    Returns
    -------
    r : ndarray
        Scores (coefficients / loadings).
    e : ndarray
        Sparse error vector.
    """
    xp = array_namespace(z)
    m, n = X.shape
    z = z.T

    if len(z.shape) == 2:
        batch_size = z.shape[1]
        eshape = (m, batch_size)
        rshape = (n, batch_size)
    else:
        eshape = (m,)
        rshape = (n,)
    if r is None or r.shape != rshape:
        r = xp.zeros(rshape, device=getattr(z, "device", None))
    if e is None or e.shape != eshape:
        e = xp.zeros(eshape, device=getattr(z, "device", None))

    # Precompute the pseudo-inverse term
    ddt = xp.linalg.solve(X.T @ X + Id, X.T)
    maxiter = int(1e6)
    itr = 0

    while True:
        itr += 1
        # Solve for scores r
        rtmp = r
        r = ddt @ (z - e)

        # Solve for sparse error e
        etmp = e
        e = _soft_thresh(z - X @ r, lambda2)

        # Check convergence
        stopr = float(xp.linalg.norm(r - rtmp))
        stope = float(xp.linalg.norm(e - etmp))
        stop = max(stopr, stope) / m
        if stop < 1e-5 or itr > maxiter:
            break

    return r, e


def _updatecol(X, A, B, Id):
    """Column-wise update of the subspace basis for the BCD solver.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Subspace basis to update in-place.
    A : ndarray, shape (n, n)
        Accumulated Gram matrix.
    B : ndarray, shape (m, n)
        Accumulated cross-term matrix.
    Id : ndarray, shape (n, n)
        Regularisation matrix.

    Returns
    -------
    L : ndarray, shape (m, n)
        Updated subspace basis.
    """
    xp = array_namespace(X)
    _, n = X.shape
    L = X
    A = A + Id

    for i in range(n):
        b = B[:, i]
        x = X[:, i]
        a = A[:, i]
        temp = (b - X @ a) / A[i, i] + x
        col_norm = max(float(xp.linalg.norm(temp)), 1.0)
        L[:, i] = temp / col_norm

    return L


class ORPCA:
    """Online Robust Principal Component Analysis.

    Decomposes a data matrix into low-rank and sparse components using
    online stochastic optimisation. The model is updated incrementally
    as new samples arrive, making it suitable for streaming data and
    datasets that do not fit in memory.

    The algorithm is based on [Feng2013]_ with extensions for stochastic
    gradient descent (SGD) and momentum-based optimisation [Ruder2016]_.

    Parameters
    ----------
    n_components : int
        Number of components (rank of the low-rank subspace).
    store_error : bool, default False
        If True, the sparse error matrix is stored and accessible via
        the ``sparse_`` attribute after fitting.
    lambda1 : float, default 0.1
        Nuclear-norm regularisation parameter.
    lambda2 : float, default 1.0
        Sparse-error regularisation parameter.
    method : {'CF', 'BCD', 'SGD', 'MomentumSGD'}, default 'BCD'
        Solver used for the subspace update step:

        - ``'CF'``  — Closed-form solution.
        - ``'BCD'`` — Block-coordinate descent (default).
        - ``'SGD'`` — Stochastic gradient descent.
        - ``'MomentumSGD'`` — SGD with momentum.
    init : {'qr', 'rand'} or ndarray, default 'qr'
        Subspace initialisation method:

        - ``'qr'``   — QR decomposition of the first *training_samples*.
        - ``'rand'`` — Random initialisation.
        - ndarray of shape ``(n_features, n_components)``.
    training_samples : int, default 10
        Number of samples used for ``'qr'`` initialisation.
        Must be >= ``n_components``.
    subspace_learning_rate : float, default 1.0
        Learning rate for the ``'SGD'`` and ``'MomentumSGD'`` methods.
        Must be > 0.
    subspace_momentum : float, default 0.5
        Momentum coefficient for ``'MomentumSGD'`` (between 0 and 1).
    random_state : None, int, or numpy.random.RandomState, default None
        Random seed or RandomState for reproducible results.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Learned subspace basis (rows are components — sklearn convention).
    low_rank_ : ndarray of shape (n_samples, n_features)
        Low-rank reconstruction of the fitted data.
    sparse_ : ndarray of shape (n_samples, n_features) or None
        Sparse error matrix. Only populated when ``store_error=True``.

    Notes
    -----
    The estimator is inherently online: call ``partial_fit`` repeatedly
    with new chunks of data to update the model incrementally without
    revisiting past samples.

    References
    ----------
    .. [Feng2013] Jiashi Feng, Huan Xu and Shuicheng Yuan, "Online Robust
       PCA via Stochastic Optimization", Advances in Neural Information
       Processing Systems 26, (2013), pp. 404–412.
    .. [Ruder2016] Sebastian Ruder, "An overview of gradient descent
       optimization algorithms", arXiv:1609.04747, (2016).
    """

    def __init__(
        self,
        n_components,
        store_error=False,
        lambda1=0.1,
        lambda2=1.0,
        method="BCD",
        init="qr",
        training_samples=10,
        subspace_learning_rate=1.0,
        subspace_momentum=0.5,
        random_state=None,
    ):
        self.n_components = n_components
        self.store_error = store_error
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.method = method
        self.init = init
        self.training_samples = training_samples
        self.subspace_learning_rate = subspace_learning_rate
        self.subspace_momentum = subspace_momentum
        self.random_state = random_state

        # Internal state — initialised lazily on first fit / partial_fit.
        self._L = None  # subspace basis, shape (n_features, n_components)
        self._R = []  # list of score arrays
        self._K = None  # regularisation matrix (lambda1 * I)
        self._A = None  # accumulated Gram matrix (CF, BCD)
        self._B = None  # accumulated cross-term (CF, BCD)
        self._vnew = None  # momentum accumulator (MomentumSGD)
        self._r = None  # last scores
        self._e = None  # last error
        self._E = [] if store_error else None  # error history
        self._n_features = None
        self._iterating = False
        self._t = 0  # step counter

        self._rs = _check_random_state(random_state)

        # ----- validation -----
        if method not in ("CF", "BCD", "SGD", "MomentumSGD"):
            raise ValueError("'method' not recognised")
        if not isinstance(init, np.ndarray) and init not in ("qr", "rand"):
            raise ValueError("'init' not recognised")
        if not isinstance(init, np.ndarray):
            if init == "qr" and training_samples < n_components:
                raise ValueError("'training_samples' must be >= 'n_components'")
        if method == "MomentumSGD" and (
            subspace_momentum > 1.0 or subspace_momentum < 0.0
        ):
            raise ValueError("'subspace_momentum' must be a float between 0 and 1")

    # ------------------------------------------------------------------
    # sklearn-compatible public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the online RPCA model to *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or iterable
            Training data. May be an array or an iterator that yields
            individual samples (each a 1-D array of length n_features).
        y : Ignored
            Not used; present for API consistency.

        Returns
        -------
        self : ORPCA
            Fitted estimator.
        """
        self._reset_state()
        self._fit_impl(X, batch_size=None)
        return self

    def transform(self, X):
        """Project *X* onto the learned components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            Coordinates of each sample in the learned subspace.
        """
        return self._project_impl(X).T

    def fit_transform(self, X, y=None):
        """Fit the model and return the scores for *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or iterable
            Training data.
        y : Ignored
            Not used; present for API consistency.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_components)
            Scores (projections) of the training data onto the learned
            components.
        """
        self.fit(X)
        return self.transform(X)

    def partial_fit(self, X, batch_size=None):
        """Process one batch of data, updating the model incrementally.

        Call this repeatedly with successive chunks of data for true
        online / streaming learning.  The first call initialises the
        internal state; subsequent calls update the existing model.

        Parameters
        ----------
        X : ndarray of shape (n_chunk, n_features)
            A chunk of observations.
        batch_size : int or None
            If not None, split *X* into sub-batches of this size
            before processing.

        Returns
        -------
        self : ORPCA
            Updated estimator.
        """
        self._fit_impl(X, batch_size=batch_size)
        return self

    # ------------------------------------------------------------------
    # sklearn-compatible attributes
    # ------------------------------------------------------------------

    @property
    def components_(self):
        """Learned subspace, shape ``(n_components, n_features)`` — sklearn
        convention (rows are components)."""
        if self._L is None:
            raise ValueError(
                "Model has not been fitted yet. Call fit() or partial_fit() first."
            )
        return self._L.T

    @property
    def low_rank_(self):
        """Low-rank reconstruction of the fitted data,
        shape ``(n_samples, n_features)``."""
        if self._L is None or len(self._R) == 0:
            raise ValueError(
                "Model has not been fitted yet. Call fit() or partial_fit() first."
            )
        if len(self._R[0].shape) == 1:
            R = np.stack(self._R, axis=-1)
        else:
            R = np.concatenate(self._R, axis=1)
        return (self._L @ R).T

    @property
    def sparse_(self):
        """Sparse error matrix, shape ``(n_samples, n_features)``.

        Returns ``None`` unless ``store_error=True`` was set at
        construction time.
        """
        if self._E is None or len(self._E) == 0:
            return None
        return np.array(self._E).T

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        """Discard any previously-learned model state."""
        self._L = None
        self._R = []
        self._K = None
        self._A = None
        self._B = None
        self._vnew = None
        self._r = None
        self._e = None
        self._E = [] if self.store_error else None
        self._n_features = None
        self._iterating = False
        self._t = 0

    def _setup(self, X):
        """Initialise internal structures from the first data batch.

        Returns the (possibly wrapped) iterator over *X*.
        """
        self._r, self._e = None, None
        if isinstance(X, np.ndarray):
            n, m = X.shape
            iterating = False
        else:
            x = next(X)
            m = len(x)
            X = chain([x], X)
            iterating = True

        self._n_features = m
        self._iterating = iterating

        self._L = self._initialise_subspace(X)
        self._K = self.lambda1 * np.eye(self.n_components)
        self._R = []

        if self.method in ("CF", "BCD"):
            self._A = np.zeros((self.n_components, self.n_components))
            self._B = np.zeros((m, self.n_components))
        elif self.method == "MomentumSGD":
            self._vnew = np.zeros_like(self._L)

        return X

    def _initialise_subspace(self, X):
        """Return an initial subspace basis of shape (n_features, n_components)."""
        m = self._n_features

        if isinstance(self.init, np.ndarray):
            if self.init.ndim != 2:
                raise ValueError("'init' must be a 2-D matrix")
            init_m, init_r = self.init.shape
            if init_m != m or init_r != self.n_components:
                raise ValueError("'init' must have shape (n_features, n_components)")
            return self.init.copy()
        elif self.init == "qr":
            if self._iterating:
                Y2 = np.stack([next(X) for _ in range(self.training_samples)], axis=-1)
                X = chain(iter(Y2.T.copy()), X)
            else:
                Y2 = X[: self.training_samples, :].T
            L, _ = scipy.linalg.qr(Y2, mode="economic")
            return L[:, : self.n_components]
        elif self.init == "rand":
            Y2 = self._rs.normal(size=(m, self.n_components))
            L, _ = scipy.linalg.qr(Y2, mode="economic")
            return L[:, : self.n_components]

    def _fit_impl(self, X, batch_size=None):
        """Internal fit logic shared by :meth:`fit` and :meth:`partial_fit`."""
        if self._L is None:
            X = self._setup(X)

        num = None
        if batch_size is not None:
            if not isinstance(X, np.ndarray):
                raise ValueError("Cannot batch iterating data")
            length = X.shape[0]
            num = max(length // batch_size, 1)
            X = np.array_split(X, num, axis=0)

        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)

        r, e = self._r, self._e

        for v in tqdm(X, leave=False, total=num, disable=num == 1):
            r, e = _solveproj(v, self._L, self._K, self.lambda2, r=r, e=e)
            self._r = r
            self._e = e
            self._R.append(np.asarray(r))
            if self._E is not None:
                self._E.append(np.asarray(e))

            # Compute outer / inner products for the subspace update.
            r_np = np.asarray(r)
            v_np = np.asarray(v)
            e_np = np.asarray(e)
            if r_np.ndim == 1:
                A_update = np.outer(r_np, r_np)
                B_update = np.outer(v_np.T - e_np, r_np)
            else:
                A_update = r_np @ r_np.T
                B_update = (v_np.T - e_np) @ r_np.T
            self._solve_L(A_update, B_update)
            self._t += 1

    def _solve_L(self, A, B):
        """Update the subspace basis ``_L`` using the chosen solver."""
        if self.method == "CF":
            self._A += A
            self._B += B
            self._L = self._B @ np.linalg.inv(self._A + self._K)
        elif self.method == "BCD":
            self._A += A
            self._B += B
            self._L = _updatecol(self._L, self._A, self._B, self._K)
        elif self.method == "SGD":
            learn = self.subspace_learning_rate * (
                1.0 + self.subspace_learning_rate * self.lambda1 * self._t
            )
            self._L -= (self._L @ A - B + self.lambda1 * self._L) / learn
        elif self.method == "MomentumSGD":
            learn = self.subspace_learning_rate * (
                1.0 + self.subspace_learning_rate * self.lambda1 * self._t
            )
            vold = self.subspace_momentum * self._vnew
            self._vnew = (self._L @ A - B + self.lambda1 * self._L) / learn
            self._L -= vold + self._vnew

    def _project_impl(self, X, return_error=False):
        """Project data onto the current subspace.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or iterable
            Data to project.
        return_error : bool
            If True, also return the sparse error.

        Returns
        -------
        scores : ndarray of shape (n_components, n_samples)
            Scores (coefficients).
        errors : ndarray, optional
            Sparse error matrix (only if *return_error* is True).
        """
        R = []
        E = [] if return_error else None

        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)

        for v in tqdm(X, leave=False, total=num):
            r, e = _solveproj(v, self._L, self._K, self.lambda2)
            R.append(np.asarray(r).copy())
            if return_error:
                E.append(np.asarray(e).copy())

        R = np.stack(R, axis=-1)
        if return_error:
            return R, np.stack(E, axis=-1)
        return R


class RPCAGoDec:
    """Robust PCA via the GoDec algorithm (batch).

    Decomposes a data matrix into low-rank and sparse components using
    bilateral random projections.  This is a batch method — it processes
    the entire dataset at once and is best suited for data that fit in
    memory.

    The algorithm is based on [Zhou2011]_.

    Parameters
    ----------
    rank : int, default 5
        Target rank of the low-rank approximation.
    lambda1 : float or None, default None
        Threshold for soft-thresholding the sparse error.
        If None, defaults to ``1 / sqrt(n_features)``.
    power : int, default 0
        Number of power iterations used during randomised initialisation.
    tol : float, default 1e-3
        Convergence tolerance on the Frobenius norm of the residual.
    max_iter : int, default 100
        Maximum number of iterations.
    random_state : None, int, or numpy.random.RandomState, default None
        Random seed or RandomState for reproducible initialisation.

    Attributes
    ----------
    components_ : ndarray of shape (rank, n_features)
        Right singular vectors from the final SVD of the low-rank matrix
        (sklearn convention: rows are components).
    low_rank_ : ndarray of shape (n_samples, n_features)
        Low-rank approximation of the input data.
    sparse_ : ndarray of shape (n_samples, n_features)
        Sparse error matrix.
    singular_values_ : ndarray of shape (rank,)
        Singular values from the final SVD.

    References
    ----------
    .. [Zhou2011] Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank &
       Sparse Matrix Decomposition in Noisy Case", ICML-11, (2011), pp. 33–40.
    """

    def __init__(
        self,
        rank=5,
        lambda1=None,
        power=0,
        tol=1e-3,
        max_iter=100,
        random_state=None,
    ):
        self.rank = rank
        self.lambda1 = lambda1
        self.power = power
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the GoDec RPCA model to *X*.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used; present for API consistency.

        Returns
        -------
        self : RPCAGoDec
            Fitted estimator.
        """
        X = np.asarray(X)
        self._fit_godec(X)
        return self

    def transform(self, X):
        """Project *X* onto the learned components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        scores : ndarray of shape (n_samples, rank)
            Scores (projections) of *X* in the component space.
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X)
        return X @ self.components_.T

    def fit_transform(self, X, y=None):
        """Fit the model and return the low-rank reconstruction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used; present for API consistency.

        Returns
        -------
        low_rank : ndarray of shape (n_samples, n_features)
            Low-rank reconstruction of the training data.
        """
        self.fit(X)
        return self.low_rank_

    @property
    def components_(self):
        """Right singular vectors, shape ``(rank, n_features)``.

        Returns ``None`` if the model has not been fitted.
        """
        if not hasattr(self, "_components"):
            return None
        return self._components

    @components_.setter
    def components_(self, value):
        self._components = value

    @property
    def low_rank_(self):
        """Low-rank reconstruction, shape ``(n_samples, n_features)``."""
        if not hasattr(self, "_low_rank"):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._low_rank

    @low_rank_.setter
    def low_rank_(self, value):
        self._low_rank = value

    @property
    def sparse_(self):
        """Sparse error matrix, shape ``(n_samples, n_features)``."""
        if not hasattr(self, "_sparse"):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._sparse

    @sparse_.setter
    def sparse_(self, value):
        self._sparse = value

    @property
    def singular_values_(self):
        """Singular values from the final SVD, shape ``(rank,)``."""
        if not hasattr(self, "_singular_values"):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._singular_values

    @singular_values_.setter
    def singular_values_(self, value):
        self._singular_values = value

    # ------------------------------------------------------------------
    # internal implementation
    # ------------------------------------------------------------------

    def _fit_godec(self, X):
        """Core GoDec algorithm.

        *X* has shape ``(n_samples, n_features)`` (sklearn convention).
        Internally we transpose to ``(n_features, n_samples)`` because
        the algorithm is optimised for ``n_features >= n_samples``.
        """
        # Transpose to (n_features, n_samples) for the algorithm.
        Xt = X.T
        xp = array_namespace(Xt)
        m, n = Xt.shape

        # Operate on the transposed matrix for speed when m < n.
        transpose = m < n
        if transpose:
            Xt = Xt.T
            m, n = n, m

        if self.lambda1 is None:
            _logger.info("Threshold 'lambda1' set to default: 1 / sqrt(n_features)")
            lambda1 = 1.0 / np.sqrt(float(m))
        else:
            lambda1 = self.lambda1

        L = Xt
        E = xp.zeros(L.shape, device=getattr(Xt, "device", None))

        random_state = _check_random_state(self.random_state)

        for itr in range(int(self.max_iter)):
            # Bilateral random projection
            Y2 = random_state.normal(size=(n, self.rank))
            for _ in range(self.power + 1):
                Y2 = L.T @ (L @ Y2)

            Q, _ = scipy.linalg.qr(np.asarray(Y2), mode="economic")

            # Estimate low-rank and sparse components
            Lnew = (L @ Q) @ Q.T
            A = L - Lnew + E
            L = Lnew
            E = _soft_thresh(A, lambda1)
            A = A - E
            L = L + A

            eps = float(xp.linalg.norm(A))
            if eps < self.tol:
                _logger.info(f"Converged to {eps:.2e} in {itr} iterations")
                break

        if transpose:
            L = L.T
            E = E.T

        Xhat = L
        Ehat = E

        # Final SVD (scipy operates on numpy arrays).
        Xhat_np = np.asarray(Xhat)
        U, S, Vh = scipy.linalg.svd(Xhat_np, full_matrices=False)

        # Truncate and zero-out numerical noise.
        S[self.rank :] = 0.0

        # Store results in sklearn convention: (n_samples, n_features).
        # Xhat has shape (n_features, n_samples); its SVD is U @ diag(S) @ Vh.
        # The right singular vectors of Xhat.T (n_samples, n_features) are U.T,
        # so components_ = U.T[:rank, :] has shape (rank, n_features).
        self.low_rank_ = Xhat_np.T
        self.sparse_ = np.asarray(Ehat).T
        self.components_ = U.T[: self.rank, :]
        self.singular_values_ = S[: self.rank]
