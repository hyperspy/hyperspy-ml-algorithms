====================================
hyperspy-ml-algorithms Documentation
====================================

**hyperspy-ml-algorithms** provides standalone, sklearn-like machine learning
algorithms for signal decomposition and transformation, extracted from
`HyperSpy <https://hyperspy.org>`__.

These estimators are self-contained — they have **no dependency on HyperSpy**
itself and work with standard NumPy arrays. They are designed for
hyperspectral data analysis but can be used for any multi-dimensional dataset.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user_guide/index
   reference/index

.. grid:: 1 2 2 2

   .. grid-item-card::
      :link: user_guide/index
      :link-type: doc

      :fas:`book` **User Guide**
      ^^^
      Installation, quick start, and detailed examples for every estimator.

   .. grid-item-card::
      :link: reference/index
      :link-type: doc

      :fas:`code` **API Reference**
      ^^^
      Complete API documentation with constructors, methods, and attributes.

   .. grid-item-card::
      :link: https://github.com/hyperspy/hyperspy-ml-algorithms
      :link-type: url

      :fas:`code-branch` **GitHub Repository**
      ^^^
      Source code, issue tracker, and contributing guidelines.
