# pisco.jl

https://github.com/fmrifrey/pisco.jl

[![docs-stable][docs-stable-img]][docs-stable-url]
[![docs-dev][docs-dev-img]][docs-dev-url]
[![action][action-img]][action-url]
<!-- [![Aqua QA][aqua-img]][aqua-url] -->
<!-- [![codecov][codecov-img]][codecov-url] -->
<!-- [![deps][deps-img]][deps-url] -->
[![license][license-img]][license-url]
<!-- [![pkgeval][pkgeval-img]][pkgeval-url] -->
<!-- [![version][ver-img]][ver-url] -->

This Julia language repo provides methods for estimating sensitivity and spatiotemporal maps in MRI,
written by David Frey (djfrey@umich.edu), based on similar [matlab-based software package](github.com/ralobos/PISCO) by Rodrigo Lobos (rlobos@umich.edu)

## Getting started


## Citations

The problem formulation and methods implemented by the associated software to this script were originally reported in:

- **[1]** R. A. Lobos, C.-C. Chan, J. P. Haldar. *New Theory and Faster Computations for Subspace-Based Sensitivity Map Estimation in Multichannel MRI*. IEEE Transactions on Medical Imaging 43:286-296, 2024.
- **[2]** R. A. Lobos, C.-C. Chan, J. P. Haldar. *Extended Version of "New Theory and Faster Computations for Subspace-Based Sensitivity Map Estimation in Multichannel MRI"*, 2023, arXiv:2302.13431. ([https://arxiv.org/abs/2302.13431](https://arxiv.org/abs/2302.13431))
- **[3]** R. A. Lobos, X. Wang, R. T. L. Fung, Y. He, D. Frey, D. Gupta, Z. Liu, J. A. Fessler, D. C. Noll. *Spatiotemporal Maps for Dynamic MRI Reconstruction*, 2025, arXiv:2507.14429. ([https://arxiv.org/abs/2507.14429](https://arxiv.org/abs/2507.14429))

## Compatibility

Tested with Julia ≥ 1.12.

<!-- URLs -->
[action-img]: https://github.com/fmrifrey/pisco.jl/workflows/CI/badge.svg
[action-url]: https://github.com/fmrifrey/pisco.jl/actions

[aqua-img]: https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[code-blue-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[code-blue-url]: https://github.com/invenia/BlueStyle

[codecov-img]: https://codecov.io/github/fmrifrey/pisco.jl/coverage.svg
[codecov-url]: https://codecov.io/github/fmrifrey/pisco.jl

[deps-img]: https://juliahub.com/docs/pisco/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/pisco

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://fmrifrey.github.io/pisco.jl/dev
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://fmrifrey.github.io/pisco.jl/stable

[license-img]: https://img.shields.io/badge/license-MIT-brightgreen.svg
[license-url]: LICENSE

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/M/pisco.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/t/pisco.html

[ver-img]: https://juliahub.com/docs/pisco/version.svg
[ver-url]: https://juliahub.com/ui/Packages/pisco