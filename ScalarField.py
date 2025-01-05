import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.special as sp

r_range=.5e-5
color_range=2
resolution=0.01
bessel_n=40
num_frames = 5
animation_speed=60 #lower is faster

f=4e14
c=2.998e8
theta_k = np.pi*0 #angle of wave
a=.1e-5 # radius of cylinder
n_p=1 # index outside of cylinder
n_m=0.8 # index of cylinder
omega=(2*np.pi*f)
k_p=(omega*n_p)/c
k_m=(omega*n_m)/c
T=1/f

t_range=np.linspace(0, T-(T/num_frames), num_frames)
r = np.arange(0, r_range, resolution*r_range)
theta = np.arange(0, 2*np.pi, resolution)
r_grid, theta_grid = np.meshgrid(r, theta)

#Scalar Wave INPUT bessel, hankel coefficients
A_M=[1j**M for M in range(-bessel_n,bessel_n+1)]
B_M_m=np.zeros((2*bessel_n+1), complex)
B_M_p=np.zeros((2*bessel_n+1), complex)

# Alpha function
def alpha(M,negative,hankelfirst,hankelsecond):
    nka_p=n_p*k_p*a
    nka_m=n_m*k_m*a
    if negative:
        result=n_m
        result *= sp.h1vp(M,nka_p) if hankelfirst else sp.jvp(M,nka_p)  #alternates positive
        result *= sp.hankel1(M,nka_m) if hankelsecond else sp.jv(M,nka_m)
    else: #positive
        result=n_p
        result *= sp.h1vp(M,nka_m) if hankelfirst else sp.jvp(M,nka_m)  #alternates negative
        result *= sp.hankel1(M,nka_p) if hankelsecond else sp.jv(M,nka_p)
    return result

#SCATTER MATRIX
R_EE_m = np.zeros((2*bessel_n+1), complex)
R_EE_p = np.zeros((2*bessel_n+1), complex)
for M in range(-bessel_n,bessel_n+1):
    delta_M=(alpha(M,1,1,0)-alpha(M,0,0,1))*(n_m**2*alpha(M,0,0,1)-n_p**2*alpha(M,1,1,0))
    R_EE_m[M+bessel_n]=(alpha(M,0,0,1)-alpha(M,1,1,0))*(n_m**2*alpha(M,0,1,1)-n_p**2*alpha(M,1,1,1))/delta_M
    R_EE_p[M+bessel_n]=(alpha(M,0,0,1)-alpha(M,1,1,0))*(n_m**2*alpha(M,0,0,0)-n_p**2*alpha(M,1,0,0))/delta_M

B_M_m=np.divide(A_M,R_EE_m)   #negative R for inside a?
B_M_p=np.multiply(R_EE_p,A_M)  #positive R for outside a?
#A_M=np.zeros((2*bessel_n+1), complex)  #remove incoming wave


#SUM HANKELS AND BESSELS
def compute_field(t):
    Z = np.zeros((len(theta), len(r)), complex)
    for Q in range(-bessel_n,bessel_n+1):
        Z += (A_M[Q+bessel_n]*sp.jv(Q, n_p*k_p*r_grid)+B_M_p[Q+bessel_n]*sp.hankel1(Q,n_p*k_p*r_grid))*np.exp(1j*Q*(theta_grid-theta_k))
    Z *= np.exp(-1j*omega*t)
    return np.real(Z)
    #return np.real(np.exp(1j*(n_p*k_p*r_grid*np.cos(theta_grid-theta_k)-omega*t))) #plane wave

frames = [compute_field(i) for i in t_range]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c=ax.pcolormesh(theta_grid, r_grid, frames[0], cmap='gist_heat', vmax=color_range,vmin=-color_range)
fig.colorbar(c, ax=ax, label='Scalar Value')
ax.set_rmax(r_range)
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_title("Polar Plot of scalar field V(r,theta)", va='bottom')
def update(frame):
    c.set_array(frames[frame].flatten())
    return c,
ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=animation_speed, blit=True)
plt.show()
