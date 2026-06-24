==========
User Guide
==========

.. _installation:

Installation
============

hyperspy-ml-algorithms is available on PyPI::

    pip install hyperspy-ml-algorithms

To install with GPU support (optional)::

    pip install hyperspy-ml-algorithms[gpu]

To install with scikit-learn support (optional, enables randomized SVD)::

    pip install hyperspy-ml-algorithms[sklearn]

Quick Start
===========

All estimators follow a scikit-learn-compatible API with ``fit``,
``transform``, and ``fit_transform`` methods::

    import numpy as np
    from hyperspy_ml_algorithms import SVDPCA

    rng = np.random.RandomState(42)
    data = rng.random((77, 13))   # 77 samples, 13 features

    est = SVDPCA(n_components=5)
    est.fit(data)

    print(est.components_.shape)  # (5, 13) — rows are components
    scores = est.transform(data)
    print(scores.shape)           # (77, 5) — rows are samples

Estimator Overview
==================

The package provides 8 estimators covering a range of decomposition and
transformation techniques:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Estimator
     - Type
     - Key Feature
   * - :class:`~hyperspy_ml_algorithms.SVDPCA`
     - SVD-based PCA
     - Multi-backend SVD with flexible centering
   * - :class:`~hyperspy_ml_algorithms.MLPCA`
     - Maximum Likelihood PCA
     - Handles heteroskedastic (Poisson) noise
   * - :class:`~hyperspy_ml_algorithms.ORPCA`
     - Online Robust PCA
     - Streaming decomposition with sparse outlier handling
   * - :class:`~hyperspy_ml_algorithms.RPCAGoDec`
     - Batch Robust PCA
     - GoDec algorithm: fast low-rank + sparse decomposition
   * - :class:`~hyperspy_ml_algorithms.ORNMF`
     - Online Robust NMF
     - Non-negative decomposition for streaming data
   * - :class:`~hyperspy_ml_algorithms.IncrementalSVD`
     - Incremental SVD
     - Streaming SVD without centering
   * - :class:`~hyperspy_ml_algorithms.Orthomax`
     - Orthomax Rotation
     - Rotation of components (Varimax when ``gamma=1.0``)
   * - :class:`~hyperspy_ml_algorithms.Whitening`
     - Whitening Transformation
     - Decorrelation via PCA or ZCA whitening

GPU Support
===========

The estimators use ``array_api_compat`` internally, which enables GPU
acceleration with CuPy or PyTorch without any code changes::

    import numpy as np
    import cupy as cp
    from hyperspy_ml_algorithms import SVDPCA

    # Generate data on GPU
    data_gpu = cp.asarray(np.random.random((77, 13)))

    est = SVDPCA(n_components=5)
    est.fit(data_gpu)               # Uses CuPy for SVD

    scores_gpu = est.transform(data_gpu)
    scores = cp.asnumpy(scores_gpu)  # Back to NumPy if needed

.. note::

   Not all estimators support all array backends. The SVD-based algorithms
   (SVDPCA, MLPCA) have the best GPU support. ORNMF and RPCAGoDec operate
   primarily on NumPy arrays.

Estimator Gallery
=================

SVDPCA
------

SVD-based PCA with flexible centering, auto-transposition, and multi-backend
support::

    import numpy as np
    from hyperspy_ml_algorithms import SVDPCA

    rng = np.random.RandomState(42)
    data = rng.random((77, 13))

    est = SVDPCA(n_components=3, centre="features")
    scores = est.fit_transform(data)
    print(f"Components: {est.components_.shape}")  # (3, 13)
    print(f"Scores: {scores.shape}")               # (77, 3)
    print(f"Explained variance ratio: {est.explained_variance_ratio_}")

MLPCA
-----

Maximum Likelihood PCA for data with known per-element variance (e.g., Poisson
noise in electron microscopy)::

    import numpy as np
    from hyperspy_ml_algorithms import MLPCA

    rng = np.random.RandomState(42)
    data = rng.poisson(15, size=(50, 30)).astype(float)
    variance = data.copy()  # Poisson: variance = mean

    est = MLPCA(n_components=4, tol=1e-8)
    est.fit(data, variance)          # variance required!
    print(f"Scores: {est.scores_.shape}")       # (50, 4)
    print(f"Components: {est.components_.shape}")  # (4, 30), like sklearn

.. warning::

   MLPCA's ``fit()`` requires a second argument ``variance`` (not ``y=None``).
   The ``components_`` attribute follows the sklearn convention, with shape
   ``(n_components, n_features)``.

ORPCA
-----

Online Robust PCA for streaming data with sparse outlier handling::

    import numpy as np
    from hyperspy_ml_algorithms import ORPCA

    rng = np.random.RandomState(42)
    data = rng.random((200, 25))

    est = ORPCA(n_components=5, method="SGD",
                subspace_learning_rate=0.5, subspace_momentum=0.9)
    # Feed data in chunks for streaming
    n_batches = 4
    for chunk in np.array_split(data, n_batches):
        est.partial_fit(chunk)

    print(f"Low-rank: {est.low_rank_.shape}")  # (200, 25)
    print(f"Components: {est.components_.shape}")  # (5, 25)

RPCAGoDec
---------

Batch Robust PCA using bilateral random projections for fast decomposition::

    import numpy as np
    from hyperspy_ml_algorithms import RPCAGoDec

    rng = np.random.RandomState(42)
    data = rng.random((150, 30))

    est = RPCAGoDec(rank=6, tol=1e-3, max_iter=50)
    low_rank = est.fit_transform(data)   # returns low_rank_, NOT scores
    print(f"Low-rank: {low_rank.shape}")      # (150, 30)
    print(f"Sparse: {est.sparse_.shape}")     # (150, 30)
    scores = est.transform(data)              # (150, 6)

.. note::

   ``RPCAGoDec.fit_transform()`` returns the *low-rank reconstruction*, not
   scores. Call ``transform()`` separately to get score projections.

ORNMF
-----

Online Robust NMF: non-negative decomposition with sparse outlier rejection::

    import numpy as np
    from hyperspy_ml_algorithms import ORNMF

    rng = np.random.RandomState(42)
    data = np.abs(rng.random((80, 20)))  # non-negative data

    est = ORNMF(n_components=5, lambda1=0.5)
    est.fit(data)
    scores = est.transform(data)
    print(f"Components: {est.components_.shape}")  # (5, 20)
    print(f"Scores: {scores.shape}")               # (80, 5)
    print(f"Components non-negative: {(est.components_ >= 0).all()}")
    print(f"Scores non-negative: {(scores >= 0).all()}")

IncrementalSVD
--------------

Streaming SVD for out-of-core data — no centering applied::

    import numpy as np
    from hyperspy_ml_algorithms import IncrementalSVD

    rng = np.random.RandomState(42)
    data = rng.random((250, 15))

    est = IncrementalSVD(n_components=4)
    # Feed in chunks
    for chunk in np.array_split(data, 5):
        est.partial_fit(chunk)

    scores = est.transform(data)
    print(f"Components: {est.components_.shape}")         # (4, 15)
    print(f"Singular values: {est.singular_values_}")     # (4,)
    print(f"Samples seen: {est.n_samples_seen_}")         # 250
    print(f"Noise variance: {est.noise_variance_}")

Orthomax
--------

Rotation of a pre-computed component matrix. The input is ``(n_features,
n_components)`` — this is a *rotation*, not a decomposition::

    import numpy as np
    from hyperspy_ml_algorithms import SVDPCA, Orthomax

    rng = np.random.RandomState(42)
    data = rng.random((90, 18))

    # First get components from a decomposition
    pca = SVDPCA(n_components=4).fit(data)

    # Rotate components (input is n_features × n_components)
    rotator = Orthomax(gamma=1.0)  # varimax rotation
    rotated = rotator.fit_transform(pca.components_.T)
    print(f"Rotation matrix: {rotator.rotation_matrix_.shape}")  # (4, 4)
    print(f"Rotated components: {rotated.shape}")                 # (18, 4)

.. warning::

   Orthomax expects input of shape ``(n_features, n_components)``, **not**
   ``(n_samples, n_features)``. You must pass the transposed components from
   another decomposition.

Whitening
---------

Decorrelate variables via PCA or ZCA whitening::

    import numpy as np
    from hyperspy_ml_algorithms import Whitening

    rng = np.random.RandomState(42)
    data = rng.random((65, 12))

    est = Whitening(method="ZCA")
    whitened = est.fit_transform(data)
    print(f"Whitening matrix: {est.whitening_matrix_.shape}")  # (12, 12)
    print(f"Whitened data: {whitened.shape}")                  # (65, 12)

    # Verify decorrelation
    cov = np.cov(whitened.T)
    print(f"Diagonal of covariance (should be ~1): {np.diag(cov)}")
