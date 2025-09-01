# PISCO Sensitivity Map Estimation for Julia
written by David Frey (djfrey@umich.edu), based on similar [matlab-based software package](github.com/ralobos/PISCO) by Rodrigo Lobos (rlobos@umich.edu)

The problem formulation and methods implemented by the associated software to this script were originally reported in:

- **[1]** R. A. Lobos, C.-C. Chan, J. P. Haldar. *New Theory and Faster Computations for Subspace-Based Sensitivity Map Estimation in Multichannel MRI*. IEEE Transactions on Medical Imaging 43:286-296, 2024.
- **[2]** R. A. Lobos, C.-C. Chan, J. P. Haldar. *Extended Version of "New Theory and Faster Computations for Subspace-Based Sensitivity Map Estimation in Multichannel MRI"*, 2023, arXiv:2302.13431. ([https://arxiv.org/abs/2302.13431](https://arxiv.org/abs/2302.13431))
- **[3]** R. A. Lobos, X. Wang, R. T. L. Fung, Y. He, D. Frey, D. Gupta, Z. Liu, J. A. Fessler, D. C. Noll. *Spatiotemporal Maps for Dynamic MRI Reconstruction*, 2025, arXiv:2507.14429. ([https://arxiv.org/abs/2507.14429](https://arxiv.org/abs/2507.14429))

## Getting started

Sample 2D and 3D multi-channel MRI datasets are provided in the [`/data`](./data) directory. The [`pisco_demo_2D.ipynb`](./pisco_demo_2D.ipynb) notebook offers a detailed, step-by-step walkthrough of sensitivity map estimation from the 2D sample dataset using PISCO. For 3D data, [`pisco_demo_3D.ipynb`](./pisco_demo_3D.ipynb) demonstrates how to apply the relevant functions in this package to estimate sensitivity maps, focusing on practical usage rather than step-by-step explanation.

## dev TODO list
- look into subspace iteration speed
- look into multithreading for subspace iteration/SVD on G matrix
- add more theory/documentation in 2D example markdown
- create undersampled SENSE recon examples
- create undersampled k-t STM recon examples