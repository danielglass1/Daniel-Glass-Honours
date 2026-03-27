# `mp_torch`: A Differentiable Electromagnetic Solver for Radiative Cooling Textiles

A custom, PyTorch-based multipole scattering solver designed for modeling and optimising cylindrical geometries, specifically porous polymer fibers used in passive radiative cooling.
This code was developed as part of a Physics Honours program at the University of Sydney, the accompanying thesis may be found in this repository `thesis.pdf`.

This codebase provides a fully differentiable, semi-analytical simulation environment. By wrapping SciPy's Bessel functions with custom PyTorch autograd functions, `mp_torch` allows for gradient-based optimization of complex microstructures (like hole positions and radii within a fiber) to maximize radiative cooling efficiency.

## Repository Structure

* `mp_torch/` - The core solver package.
  * `solver.py`: The main solver class that handles the multipole expansion, Wiscombe truncation, and matrix assemblies.
  * `torch_bessel.py`: Custom PyTorch autograd classes (`BesselJv`, `BesselYv`) that provide exact derivatives for SciPy's complex Bessel functions.
  * `get_matrices.py`: Computes reflection and transmission scattering matrices for the jacket and internal cylinders.
  * `generate_field.py`: Reconstructs the internal and scattered 2D field maps from the solved coefficients.
  * `forward.py`: Calculates the forward scattering cross-section (the primary Figure of Merit for optimization).
* `optimiser.py`: The gradient-based optimization script to generate new fiber geometries.
* `plotting_example.py`: A lightweight example script demonstrating how to define a structure, run the solver, and plot the fields.

## Requirements

Ensure you have a Python environment with the following dependencies:
* `torch`
* `scipy`
* `matplotlib`

## Getting Started

### 1. Running a Basic Simulation

You can define a fiber (the "jacket") and a series of holes (the "cylinders"), and solve for the internal and scattered fields. Check `plotting_example.py` for a complete implementation.

### 2. Optimizing a Fiber Geometry
To generate new, optimized geometries using gradient descent, run 'optimiser.py':

The optimizer will:

Initialize a random configuration of non-overlapping holes inside the fiber using circle packing.

Evaluate the forward scattering objective across a defined spectrum of wavenumbers (k_0).

Use backpropagation via AdamW to shift hole positions and adjust radii to minimize the objective function.

Save the optimized geometries as .pt tensor files and render preview images of the cross-section.

## Author
Daniel Glass
