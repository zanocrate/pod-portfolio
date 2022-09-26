# Physics of Data portfolio

This repository stores most of the projects and homeworks I have worked on during my studies at the University of Padova for the Physics of Data master degree. The repository is organized in directories, each directory being a course in the curriculum.

Starting with the most important one:

## `LCP/`: Laboratory of Computational Physics repository

The two projects of the annual course of Laboratory of Computational Physics are stored in this subdirectory, along with several small homeworks assigned during the year.

### `LCP/wavelets/`

This project revolves around the use of Wavelets transformation to perform a time-frequency analysis of mechanical sensors and detect the respiratory rate of a human subject.

The main noteboot is available in `./wavelets/Project.ipynb`
### `LCP/SBAM/`

The goal of this project was to develop a module compatible with [PySindy](https://github.com/dynamicslab/pysindy)'s environment for data-driven model discovery of non linear dynamics.

The module is intended to work with the rest of `PySINDy` library and provides a different approach to the linear regression component of the algorithm, implementing a hierarchical Bayesian model that integrates a Spike and Slab prior in order to model the sparsity hypothesis that makes the problem so peculiar.

The module code is available in `./SBAM/module.ipynb`, while tests of its performance are available in `./SBAM/experiments.ipnyb`, and finally a slideshow presentation of the whole project is compiled in `./SBAM/SBAM.pdf`


## `MAPD/` : Management and Analysis of Physical Datasets

The goal of this project was to perform an analysis of a large number of COVID-19 related papers, stored in `.json` unstructured format, using parallel computing resources provided by the CloudVeneto infrastructure and Python Dask libraries. 

The Jupyter Notebook was compiled on the remote cluster and is available in `MAPD/Project_Gr9bis.ipynb`.

## `AdvStat` : Advanced Statistics

The course revolved around Bayesian statistics and the use of R as a programming language to perform statistical analysis. The homeworks assigned for this are available as Jupter Notebooks under the several `AdvStat/2057447lab*` directories.

## `ML` : machine learning

The machine learning course assignments as Jupyter Notebooks are available in `ML/` subdirectories and revolve around the use of ML basic models such as SVM or Neural Networks over classic datasets such as the MNIST one.
