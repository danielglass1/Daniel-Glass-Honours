# `mp_torch`: Differentiable Electromagnetic Solver for Radiative Cooling Textiles

A custom, PyTorch-based multipole scattering solver designed for modeling and optimizing cylindrical geometries, specifically porous polymer fibers used in passive radiative cooling. 

This codebase provides a fully differentiable, semi-analytical simulation environment. By wrapping SciPy's Bessel functions with custom PyTorch autograd functions, `mp_torch` allows for gradient-based optimization of complex microstructures (like hole positions and radii within a fiber) to maximize radiative cooling efficiency.

## Features

* **Multipole Scattering Solver:** Solves for the electromagnetic fields of multiple interacting cylinders within a larger cylindrical jacket using a semi-analytical multipole expansion approach.
* **Fully Differentiable:** Implements custom `torch.autograd` wrappers for Bessel and Hankel functions, enabling seamless backpropagation through the complex coordinates of the physical simulation.
* **Gradient-Based Optimization:** Includes an optimization pipeline (`optimiser.py`) using `AdamW` to algorithmically discover highly efficient fiber geometries, incorporating spatial penalties to prevent overlapping or out-of-bounds features.
* **Automated Packing Initialization:** Uses circle-packing algorithms to generate valid, randomized initial states for the optimizer.
* **Field Visualization:** Generates and plots high-resolution 2D distributions of both Electric (E) and Magnetic (K) fields.
* **GPU Ready:** Built entirely on PyTorch tensor operations, allowing for easy porting to CUDA devices for accelerated batch processing and optimization.

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

```python
import torch
import matplotlib.subplots as plt
from mp_torch.solver import solver
from mp_torch.generate_field import generate_field

# Define incident wave parameters
inc_k_0 = torch.tensor(2 * torch.pi) / 1.0  # Wavenumber (lambda = 1)
inc_phi = torch.tensor(0.5 * torch.pi)      # Oblique angle

# Define fiber (jacket)
fibre_a = torch.tensor(5.0)  # Radius
fibre_n_real = 1.5           # Refractive index

# Define holes: [X, Y, Radius, Re(n), Im(n)]
holes = torch.tensor([
    [-1.0, -3.0, 1.0, 1.0, 0.0],
    [-3.0,  1.0, 1.2, 1.0, 0.0],
    [ 2.0,  2.0, 2.0, 1.0, 0.0],
], dtype=torch.float32)

# Initialize and solve
sim = solver(holes, inc_k_0, inc_phi, fibre_a, fibre_n_real, 0.0, modify_trunc=1.0)

# Generate Field maps
E_field, K_field = generate_field(sim, inc_magnitude=1, inc_delta=torch.tensor(0.25*torch.pi), inc_theta=torch.tensor(0), xy_range=fibre_a*2, npts=128)

# Plot Electric Field
plt.imshow(torch.real(E_field), origin='lower', cmap='viridis')
plt.colorbar()
plt.show()
