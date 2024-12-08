import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.special as sp

r_range=1e-5
resolution=0.01
bessel_n=50
num_frames = 8
animation_speed=50 #lower is faster

f=4e14
n_r=1
c=2.998e8
theta_k = np.pi*0.5 #angle of wave

v=c/n_r
omega=(2*np.pi*f)
k=omega/v
T=1/f

t_range=np.linspace(0, T-(T/num_frames), num_frames)
r = np.arange(r_range*1e-2, r_range, resolution*r_range)
theta = np.arange(0, 2*np.pi, resolution)
r_grid, theta_grid = np.meshgrid(r, theta)

#Z=np.exp(1j*(n_r*k*r_grid*np.cos(theta_grid-theta_k)-omega*t))

# add up bessels
def compute_field(t):
    Z = np.zeros((len(theta), len(r)), complex)
    Z += sp.jv(0,n_r*k*r)
    for n in range(1,bessel_n+1):
        Z += 2*(1j**(n))*sp.jv(n, n_r*k*r_grid)*np.cos(n*(theta_grid-theta_k))
    Z *= np.exp(-1j*omega*t)
    return np.real(Z)

# compute frames for a range of t values
frames = [compute_field(i) for i in t_range]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c=ax.pcolormesh(theta_grid, r_grid, frames[0], cmap='gist_heat', shading='auto')
fig.colorbar(c, ax=ax, label='Scalar Value')

ax.set_rmax(r_range)
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_title("Polar Plot of scalar field V(r,theta)", va='bottom')

def update(frame):
    c.set_array(frames[frame].flatten())
    return c,

ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=animation_speed, blit=True  # Adjust interval for speed
)

plt.show()
