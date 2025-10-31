from mp_torch.generate_field import generate_field
from mp_torch.solver import solver
from mp_torch.forward import forward
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib
matplotlib.use('MacOSX')  # or 'Qt5Agg', 'MacOSX', etc.

### INCIDENT FIELD VAR ####################################################################
inc_phi = torch.tensor(0.5*torch.pi)                # oblique angle
inc_theta = torch.tensor(0*torch.pi)                # azimuthal angle
inc_lam=1                                           # wavelength
inc_k_0 = torch.tensor(2*torch.pi)/inc_lam          # wavenumber
inc_k_perp=inc_k_0*torch.sin(inc_phi)               # perpendicular wavenumber
inc_magnitude=1                                     # field amplitude (V_0)
inc_delta=torch.tensor(0.25*torch.pi)               # polarization angle

### STRUCTURE ###########################################################################
# fibre                                             # n - fibre refractive index
fibre_n_real = 1.5                                  # Re(n)
fibre_n_imag= 0                                     # Im(n)
fibre_a = torch.tensor(5)                           # fibre radius


# holes
#                            X    Y    R    Re(n)    Im(n)
holes=torch.tensor([
                        [   -1,   -3,   1,      1,       0],
                        [   -3,   1,   1.2,    1,       0],
                        [    2,   2,   2,      1,       0],
                 ],dtype=torch.float32)


### GENERATE FIELD #############################################################################
bessel_trunc=1.0

              #holes, wavenumber, phi,  fibre radius, fibre Re(n), fibre Im(n), bessel_truncation)
solver1=solver(holes, inc_k_0, inc_phi,  fibre_a,    fibre_n_real,   fibre_n_imag,  bessel_trunc)
S_0=solver1.jacket_B_matrix

avg_F_theta=forward(S_0,inc_k_perp,fibre_a)         #Figure of Merit - Forward Difference
print(avg_F_theta)

### PLOTTING #############################################################################

npts=128
xy_range=fibre_a*2                 
E_field, K_field = generate_field(solver1,inc_magnitude,inc_delta,inc_theta,xy_range,npts)

plot_extent = [-xy_range, xy_range, -xy_range, xy_range]

# Plot Electric Field
im=plt.imshow(torch.real(E_field), extent=plot_extent, origin='lower', cmap='vanimo', vmax=2,vmin=-2)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.gca().add_patch(patches.Circle((0, 0), fibre_a, edgecolor='white', facecolor='none', linewidth=0.2))
for x,y,r,_,_ in holes:
    plt.gca().add_patch(patches.Circle((x, y), r, edgecolor='white', facecolor='none', linewidth=0.2))
plt.show()

# Plot Magnetic Field
# im.set_data(torch.real(K_field))
# plt.show()