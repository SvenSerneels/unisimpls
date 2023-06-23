# `unisimpls`: A Julia Package for Univariate SIMPLS regression, Jacobians and prediction intervals

This package implements univariate SIMPLS regression. SIMPLS, a "statistically inspired modification to partial least squares" \[1\] is a computationally efficient PLS algorithm. The package contains an implementation adapted from MATLAB code by Sijmen de Jong.

The package provides a class that alows to interface with univariate SIMPLS through the ScikitLearn API. Widely used ScikitLearn functions, such as `fit!`, `predict`, `transform`, as well as cross-validation routines such as `GridSearchCV` can flawlessly be applied to objects of the `SIMPLS` class.

Beyond training and prediction, this package provides unique access to case specific prediction intervals estimated using the most accurate approximation based on the SIMPLS Jacobian matrices. The Jacobians are calculated using the most efficient algorithm, first published in \[2\], which is also closely related to the work presented in \[3\]. For the sake of convenience, a preprint version of the key publication (\[2\]) is included [in this package's documentation folder](https://github.com/SvenSerneels/unisimpls/blob/master/doc/PLS03_Preprint.pdf).

Installation
------------
`]add <path to this GitHub repo>`

Examples
--------
The [Jupyter Notebook](https://github.com/SvenSerneels/unisimpls/blob/master/doc/example.ipynb) in the documentation section provides examples.


References
---------
\[1\]. S. de Jong, SIMPLS: An alternative approach to partial least squares regression, Chemometrics and Intelligent Laboratory Systems, 18 (1993), 251-263.

\[2\]  S. Serneels and P.J. Van Espen, Sample specific prediction intervals in SIMPLS, in: PLS and related methods, M. Vilares, M. Tenenhaus, P. Coelho, V. Esposito Vinzi and A. Morineau (eds.),DECISIA, Levallois Perret
(France), 2003, pp. 219-233. [Pre-print available here](https://github.com/SvenSerneels/unisimpls/blob/master/doc/PLS03_Preprint.pdf).

\[3\] S. Serneels, P. Lemberge and P.J. Van Espen, Calculation of PLS prediction intervals using efficient recurrence relations for the Jacobian matrix, Journal of Chemometrics, 18 (2004), 76-80.
