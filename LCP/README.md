## Laboratory of Computational Physics repository

The two projects of the annual course of Laboratory of Computational Physics are stored in this subdirectory, along with several small homeworks assigned during the year.

### `wavelets/`

This project revolves around the use of Wavelets transformation to perform a time-frequency analysis of mechanical sensors and detect the respiratory rate of a human subject.

The main noteboot is available in `wavelets/Project.ipynb`
### `SBAM/`

The goal of this project was to develop a module compatible with [PySindy](https://github.com/dynamicslab/pysindy)'s environment for data-driven model discovery of non linear dynamics.

The module is intended to work with the rest of `PySINDy` library and provides a different approach to the linear regression component of the algorithm, implementing a hierarchical Bayesian model that integrates a Spike and Slab prior in order to model the sparsity hypothesis that makes the problem so peculiar.

The module code is available in `SBAM/module.ipynb`, while tests of its performance are available in `SBAM/experiments.ipnyb`, and finally a slideshow presentation of the whole project is compiled in `SBAM/SBAM.pdf`


