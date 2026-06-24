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

from itertools import chain

import numpy as np
import scipy

from hyperspy_ml_algorithms.utils._array_namespace import array_namespace


def _check_random_state(seed):
    """Turn *seed* into a :class:`numpy.random.RandomState` instance.

    Replaces ``hyperspy.misc.math_tools.check_random_state`` so the
    estimator has no HyperSpy dependency.

    Parameters
    ----------
    seed : None, int, or numpy.random.RandomState
        If None, return the default RandomState.
        If int, return a new RandomState seeded with *seed*.
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


def _thresh(X, lambda1, vmax, xp):
    """Soft-thresholding with clipping.

    Parameters
    ----------
    X : ndarray
    lambda1 : float
    vmax : float
    xp : array_namespace module
    """
    res = xp.abs(X) - lambda1
    res = xp.maximum(res, xp.asarray(0.0))
    res = res * xp.sign(X)
    res = xp.clip(res, xp.asarray(-vmax), xp.asarray(vmax))
    return res


def _mrdivide(B, A):
    """Solves xB = A as per Matlab.

    Uses ``numpy.linalg.solve`` / ``lstsq`` which are not part of the
    Array API standard, so this helper always operates on numpy arrays.
    """
    if isinstance(B, np.ndarray):
        if len(B.shape) == 2 and B.shape[0] == B.shape[1]:
            # square array
            return np.linalg.solve(A.T, B.T).T
        else:
            # Set rcond default value to match numpy 1.14 default value with
            # previous numpy version
            rcond = np.finfo(float).eps * max(A.shape)
            return np.linalg.lstsq(A.T, B.T, rcond=rcond)[0].T
    else:
        return B / A


def _project(W):
    """Project *W* onto the non-negative orthant and normalise columns.

    Operates on numpy arrays (internal state).
    """
    newW = np.maximum(W, 0)
    sumsq = np.sqrt(np.sum(W**2, axis=0))
    sumsq = np.maximum(sumsq, 1)
    return _mrdivide(newW, np.diag(sumsq))


def _solveproj(v, W, lambda1, kappa, xp, h=None, e=None, vmax=None):
    """Solve the projection sub-problem for one sample.

    Parameters
    ----------
    v : ndarray
        Single sample vector (or batch), shape (n_features,) or (n_features, n_batch).
    W : ndarray
        Current dictionary, shape (n_features, n_components).
    lambda1 : float
    kappa : float
    xp : module
        Array namespace module for dispatch (e.g., numpy, array_api_compat.numpy).
    h, e : ndarray or None
        Warm-start values for scores and error.
    vmax : float or None

    Returns
    -------
    h : ndarray
        Scores for *v*.
    e : ndarray
        Sparse error for *v*.
    """
    m, n = W.shape
    v = v.T
    if vmax is None:
        vmax = float(xp.max(v))

    if len(v.shape) == 2:
        batch_size = v.shape[1]
        eshape = (m, batch_size)
        hshape = (n, batch_size)
    else:
        eshape = (m,)
        hshape = (n,)
    if h is None or h.shape != hshape:
        h = np.zeros(hshape)
    if e is None or e.shape != eshape:
        e = np.zeros(eshape)

    # Use xp for the norm; W is always numpy so wrap it
    W_xp = xp.asarray(W)
    eta = kappa / float(xp.linalg.norm(W_xp) ** 2)

    maxiter = int(1e6)
    iters = 0

    while True:
        iters += 1
        # Solve for h
        htmp = h
        h = h - eta * (W_xp.T @ (W_xp @ xp.asarray(h) + xp.asarray(e) - v))
        h = np.asarray(xp.maximum(xp.asarray(h), xp.asarray(0.0)))

        # Solve for e
        etmp = e
        e = np.asarray(_thresh(v - W_xp @ xp.asarray(h), lambda1, vmax, xp))

        # Stop conditions
        stoph = float(
            xp.linalg.norm(xp.asarray(h) - xp.asarray(htmp))
        )
        stope = float(
            xp.linalg.norm(xp.asarray(e) - xp.asarray(etmp))
        )
        stop = max(stoph, stope) / m
        if stop < 1e-5 or iters > maxiter:
            break

    return h, e


class ORNMF:
    """Online Robust NMF with missing or corrupted data.

    The ORNMF code is based on a transcription of the online proximal gradient
    descent (PGD) algorithm MATLAB code obtained from the authors of [Zhao2016]_.
    It has been updated to also include L2-normalization cost function that
    is able to deal with sparse corruptions and/or outliers slightly faster
    (please see ORPCA implementation for details). A further modification
    has been made to allow for a changing subspace W, where X ~= W H^T + E
    in the ORNMF framework.

    Read more in the :ref:`User Guide <mva.rnmf>`.

    Parameters
    ----------
    n_components : int
        The rank of the representation (number of components).
    store_error : bool, default False
        If True, stores the sparse error matrix.
    lambda1 : float, default 1.0
        Nuclear norm regularization parameter.
    kappa : float, default 1.0
        Step-size for projection solver.
    method : {``'PGD'``, ``'RobustPGD'``, ``'MomentumSGD'``}, default ``'PGD'``
        * ``'PGD'`` - Proximal gradient descent
        * ``'RobustPGD'`` - Robust proximal gradient descent
        * ``'MomentumSGD'`` - Stochastic gradient descent with momentum
    subspace_learning_rate : float, default 1.0
        Learning rate for the ``'MomentumSGD'`` method. Should be a
        float > 0.0.
    subspace_momentum : float, default 0.5
        Momentum parameter for ``'MomentumSGD'`` method, should be
        a float between 0 and 1.
    random_state : None or int or RandomState, default None
        Used to initialize the subspace on the first iteration.
        See :func:`numpy.random.default_rng` for more information.

    Attributes
    ----------
    components_ : ndarray, shape (n_components, n_features)
        Learned dictionary W^T (non-negative).
    scores_ : ndarray, shape (n_samples, n_components)
        Learned scores H^T (non-negative).  Only available after :meth:`fit`
        or :meth:`fit_transform`; **not** updated by :meth:`partial_fit`.

    References
    ----------
    .. [Zhao2016] Zhao, Renbo, and Vincent YF Tan. "Online nonnegative matrix
        factorization with outliers." Acoustics, Speech and Signal Processing
        (ICASSP), 2016 IEEE International Conference on. IEEE, 2016.

    """

    def __init__(
        self,
        n_components,
        store_error=False,
        lambda1=1.0,
        kappa=1.0,
        method="PGD",
        subspace_learning_rate=1.0,
        subspace_momentum=0.5,
        random_state=None,
    ):
        self.n_features = None
        self.iterating = False
        self.t = 0

        if store_error:
            self.E = []
        else:
            self.E = None

        self.rank = n_components  # internal name preserved for algorithm compatibility
        self.robust = False
        self.subspace_tracking = False
        self.lambda1 = lambda1
        self.kappa = kappa
        self.subspace_learning_rate = subspace_learning_rate
        self.subspace_momentum = subspace_momentum
        self.random_state = _check_random_state(random_state)

        # Check options are valid
        if method not in ("PGD", "RobustPGD", "MomentumSGD"):
            raise ValueError("'method' not recognised")

        if method == "RobustPGD":
            self.robust = True

        if method == "MomentumSGD":
            self.subspace_tracking = True
            if subspace_momentum < 0.0 or subspace_momentum > 1:
                raise ValueError("'subspace_momentum' must be a float between 0 and 1")

    # ------------------------------------------------------------------
    # sklearn-compatible API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Learn NMF components from the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matrix of observations or an iterator that yields samples,
            each with n_features elements.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self : ORNMF
            Fitted estimator.
        """
        self._fit_impl(X, batch_size=None)
        return self

    def transform(self, X):
        """Project *X* onto the learned dictionary.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray, shape (n_samples, n_components)
            Non-negative coordinates of each sample in the learned dictionary.
        """
        xp = array_namespace(X)
        result = self._project_impl(X)
        return xp.asarray(result.T)

    def fit_transform(self, X, y=None):
        """Fit to *X* then project it in one call.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        scores : ndarray, shape (n_samples, n_components)
            Non-negative coordinates.
        """
        self.fit(X)
        return self.transform(X)

    def partial_fit(self, X, batch_size=None):
        """Process one batch of data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Batch of observations.
        batch_size : int or None
            If not None, split *X* into sub-batches of this size.

        Returns
        -------
        self : ORNMF
        """
        self._fit_impl(X, batch_size=batch_size)
        return self

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    @property
    def components_(self):
        """Learned dictionary, shape ``(n_components, n_features)``.

        Non-negative — follows the sklearn convention (W^T).

        Raises
        ------
        AttributeError
            If the estimator has not been fitted yet.
        """
        if not hasattr(self, "W"):
            raise AttributeError("ORNMF has not been fitted yet. Call 'fit' first.")
        return self.W.T

    @property
    def scores_(self):
        """Learned scores, shape ``(n_samples, n_components)``.

        Non-negative — built from the internal score history (H^T).
        Only meaningful after :meth:`fit` or :meth:`fit_transform`.

        Raises
        ------
        AttributeError
            If the estimator has not been fitted yet.
        """
        if not hasattr(self, "H") or len(self.H) == 0:
            raise AttributeError("ORNMF has not been fitted yet. Call 'fit' first.")
        if len(self.H[0].shape) == 1:
            H = np.stack(self.H, axis=-1)
        else:
            H = np.concatenate(self.H, axis=1)
        return H.T

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _setup(self, X):
        """Initialise internal state from *X*."""
        self.h, self.e, self.v = None, None, None
        if isinstance(X, np.ndarray):
            n, m = X.shape
            # Use abs() because negative-mean data would produce NaN from
            # sqrt, causing the convergence check to never trigger.
            avg = np.sqrt(abs(X.mean()) / m)
            iterating = False
        else:
            x = next(X)
            m = len(x)
            avg = np.sqrt(abs(np.mean(x)) / m)
            X = chain([x], X)
            iterating = True

        self.n_features = m
        self.iterating = iterating

        self.W = scipy.stats.halfnorm.rvs(
            size=(self.n_features, self.rank), random_state=self.random_state
        )
        self.W = abs(avg * self.W / np.sqrt(self.rank))
        self.H = []

        if self.subspace_tracking:
            self.vnew = np.zeros_like(self.W)
        else:
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((self.n_features, self.rank))

        return X

    def _fit_impl(self, X, batch_size=None):
        """Internal shared by :meth:`fit` and :meth:`partial_fit`."""
        if self.n_features is None:
            X = self._setup(X)

        num = None
        prod = np.outer
        if batch_size is not None:
            if not isinstance(X, np.ndarray):
                raise ValueError("can't batch iterating data")
            else:
                prod = np.dot
                length = X.shape[0]
                num = max(length // batch_size, 1)
                X = np.array_split(X, num, axis=0)

        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)

        h, e = self.h, self.e

        for v in X:
            v_np = np.asarray(v)
            h, e = _solveproj(v_np, self.W, self.lambda1, self.kappa, np, h=h, e=e)
            self.v = v_np
            self.e = e
            self.h = h
            self.H.append(h)
            if self.E is not None:
                self.E.append(e)

            self._solve_W(prod(h, h.T), prod((v_np.T - e), h.T))
            self.t += 1

        self.h = h
        self.e = e

    def _solve_W(self, A, B):
        """Update the dictionary *W* using accumulated statistics.

        Operates entirely on numpy arrays (internal state).
        """
        if not self.subspace_tracking:
            self.A += A
            self.B += B
            eta = self.kappa / np.linalg.norm(self.A, "fro")

        if self.robust:
            # exactly as in the Zhao & Tan paper
            n = 0
            lasttwo = np.zeros(2)
            # Guard against division by zero — on the first iteration
            # lasttwo[0] is 0.
            maxiter = int(1e6)  # safety bound matching _solveproj
            while n <= 2 or (
                lasttwo[0] != 0
                and abs((lasttwo[1] - lasttwo[0]) / lasttwo[0]) > 1e-5
                and n < maxiter
            ):
                self.W -= eta * (self.W @ self.A - self.B)
                self.W = _project(self.W)
                n += 1
                lasttwo[0] = lasttwo[1]
                lasttwo[1] = 0.5 * np.trace(
                    self.W.T.dot(self.W).dot(self.A)
                ) - np.trace(self.W.T.dot(self.B))
        else:
            # Tom Furnival (@tjof2) approach
            # - copied from the ORPCA implementation
            #   of gradient descent in ./rpca.py
            if self.subspace_tracking:
                learn = self.subspace_learning_rate * (
                    1 + self.subspace_learning_rate * self.lambda1 * self.t
                )
                vold = self.subspace_momentum * self.vnew
                self.vnew = (self.W @ A - B) / learn
                self.W -= vold + self.vnew
            else:
                self.W -= eta * (self.W @ self.A - self.B)

            np.maximum(self.W, 0.0, out=self.W)
            self.W /= max(np.linalg.norm(self.W, "fro"), 1.0)

    def _project_impl(self, X):
        """Internal projection — returns scores in internal (H) layout.

        Converts input to numpy internally; the public :meth:`transform`
        wraps this and converts back via ``array_namespace``.
        """
        H = []
        if isinstance(X, np.ndarray):
            X = iter(X)
        for v in X:
            v_np = np.asarray(v)
            h, _ = _solveproj(v_np, self.W, self.lambda1, self.kappa, np, vmax=np.inf)
            H.append(h.copy())

        return np.stack(H, axis=-1)
