import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp

## custom differentiable complex bessel and hankels
@jax.custom_jvp
def j1jax(n, x):
    return sp.special.jv(n,x)
@j1jax.defjvp
def j1jvp(primals, tangents):
    n, x = primals
    _, x_dot = tangents
    primal_out = j1jax(n, x)
    tangent_out = 0.5 * (j1jax(n - 1, x) - j1jax(n + 1, x)) * x_dot
    return primal_out, tangent_out
@jax.custom_jvp
def h1jax(n, x):
    return sp.special.hankel1(n,x)
@h1jax.defjvp
def h1jvp(primals, tangents):
    n, x = primals
    _, x_dot = tangents
    primal_out = h1jax(n, x)
    tangent_out = 0.5 * (h1jax(n - 1, x) - h1jax(n + 1, x)) * x_dot
    return primal_out, tangent_out
@jax.custom_jvp
def j1pjax(n, x):
    return sp.special.jvp(n,x)
@j1pjax.defjvp
def j1pjvp(primals, tangents):
    n, x = primals
    _, x_dot = tangents
    primal_out = sp.special.jvp(n,x)
    tangent_out = 0.5 * (sp.special.jvp(n - 1, x) - sp.special.jvp(n + 1, x)) * x_dot
    return primal_out, tangent_out
@jax.custom_jvp
def h1pjax(n, x):
    return sp.special.h1vp(n,x)
@h1pjax.defjvp
def h1pjvp(primals, tangents):
    n, x = primals
    _, x_dot = tangents
    primal_out = sp.special.h1vp(n,x)
    tangent_out = 0.5 * (sp.special.h1vp(n - 1, x) - sp.special.h1vp(n + 1, x)) * x_dot
    return primal_out, tangent_out

#variables, -will integrate over omega and k_z which are independant
c=1 #natural units 
omega=1
T=2*jnp.pi/(omega)
n_out=1.5+0.01j     #index outside of cylinders, imaginary indicates loss
k0=omega/c+0j
k_out=n_out*k0
k_z=0.2*k_out
k_perp_out=jnp.sqrt(k_out**2-k_z**2)
theta_k=0.25*jnp.pi  #angle of incident plane wave
vec_k_perp_out=k_perp_out*jnp.asarray([jnp.cos(theta_k),jnp.sin(theta_k)])
V0=1/jnp.sqrt(2)    #magnitude of plane wave

## plot settings
ani_frames=4
ani_speed=75
color_range=1 #should match field max magnitude
t=np.linspace(0, T-(T/ani_frames), ani_frames)

npts=128
xy_range=10*jnp.pi
x=jnp.linspace(-xy_range,xy_range,npts)
y=jnp.linspace(-xy_range,xy_range,npts)
X, Y = jnp.meshgrid(x, y)

#Incident plane wave coefficients
s_terms=50
vs_terms=jnp.arange(-s_terms,s_terms+1) 
A_E0=V0*jnp.exp(1j*vs_terms*(jnp.pi/2-theta_k))
A_B0=A_E0

class cylinder:
    count=1
    instances=[]
    full_field=[]
    def __init__(self,position,radius,refractive_index,number_of_terms):
        self.count=cylinder.count
        cylinder.instances.append(self)
        cylinder.count+=1
        
        self.pos=position
        self.r=radius
        self.n=refractive_index
        self.terms=number_of_terms
        self.vterms=jnp.arange(-self.terms,self.terms+1) 
        order=self.vterms #i didnt want to rewrite
        ## secondary terms
        self.k=self.n*k0+0j
        self.k_perp=jnp.sqrt(self.k**2-k_z**2)
        self.tau=(k_z/(self.r*self.k_perp*k_perp_out))*(n_out**2-self.n**2)
        self.A_E=V0*jnp.exp(1j*jnp.dot(vec_k_perp_out,self.pos)+1j*self.vterms*(jnp.pi/2-theta_k))
        self.A_B=self.A_E

        self.B_E=self.S_EE(order)*self.A_E[order+self.terms]+self.S_EB(order)*self.A_B[order+self.terms] 
        cylinder.full_field.append(self.B_E)
        self.B_B=self.S_BB(order)*self.A_B[order+self.terms]+self.S_BE(order)*self.A_E[order+self.terms]
        cylinder.full_field.append(self.B_B)
                        

    def alpha(self,order, cylinder_index:bool, hankel_first:bool, hankel_second:bool):
        if cylinder_index:
            result = self.k_perp/k0 #should this be the k of the cylinder??
            result *= h1pjax(order, k_perp_out*self.r) if hankel_first else j1pjax(order, k_perp_out*self.r)
            result *= h1jax(order, self.k_perp*self.r) if hankel_second else j1jax(order, self.k_perp*self.r)
        else:
            result = k_perp_out/k0
            result *= h1pjax(order, self.k_perp*self.r) if hankel_first else j1pjax(order, self.k_perp*self.r)
            result *= h1jax(order, k_perp_out*self.r) if hankel_second else j1jax(order, k_perp_out*self.r)
        return result

    def delta(self,order):
        return (self.alpha(order,True,True,False)-self.alpha(order,False,False,True))*\
    (self.n**2*self.alpha(order,False,False,True)-n_out**2*self.alpha(order,True,True,False))+\
    (order*j1jax(order,self.k_perp*self.r)*h1jax(order,k_perp_out*self.r)*self.tau)**2
    
    def S_EE(self,order): #scatter coeffs from E to E field
        return (1/self.delta(order))*(self.alpha(order,False,False,True)-self.alpha(order,True,True,False))*\
                (self.n**2*self.alpha(order,False,False,False)-n_out**2*self.alpha(order,True,False,False))-\
                order**2*j1jax(order,k_perp_out*self.r)*h1jax(order,k_perp_out*self.r)*\
                    j1jax(order,self.k_perp*self.r)**2*self.tau**2

    def S_EB(self,order): #scatter coeffs from B to B field
        return (2*order*self.tau*self.k_perp*j1jax(order,self.k_perp*self.r)**2)/ \
        (self.delta(order)*jnp.pi*k0*self.r*k_perp_out)

    def S_BE(self,order): #scatter coeffs from E to B field
        return -n_out**2*self.S_EB(order)

    def S_BB(self,order): #scatter coeffs from E to E field
        return (1/self.delta(order))*((self.alpha(order,False,False,False)-self.alpha(order,True,False,False))*\
                (self.n**2*self.alpha(order,False,False,True)-n_out**2*self.alpha(order,True,True,False))-\
                order**2*j1jax(order,k_perp_out*self.r)*h1jax(order,k_perp_out*self.r)*\
                    j1jax(order,self.k_perp*self.r)**2*self.tau**2)

c1=cylinder(jnp.asarray([-10,10]),2,1,10)  #adding more terms seems to make the ring of death worse, #may need to upgrade to float64, i changed the color range and its still no good...
c2=cylinder(jnp.asarray([0,0]),2,1,10)
c3=cylinder(jnp.asarray([10,-10]),2,1,10)

# 2x2 SH block 
def SH_block(n,m):
    def H():return h1jax(n-m,k_perp_out*jnp.linalg.norm(d.pos-c.pos))*jnp.exp(-1j*(n-m)*jnp.arctan2(d.pos[1]-c.pos[1],d.pos[0]-c.pos[0]))
    return jnp.array([( -1*d.S_EE(n)*H() , -1*d.S_EB(n)*H() ) , (-1*d.S_BE(n)*H() , -1*d.S_BB(n)*H() )])
        
M=[]
for d in cylinder.instances:
    Mcol = []
    for c in cylinder.instances:
        if c!=d:
            n_vals = c.vterms
            m_vals = d.vterms
            SH=[]
            for m in m_vals:
                SHcol=[]
                for n in n_vals:
                    SHcol.append(SH_block(n,m))
                SH.append(jnp.vstack(SHcol))
            Mcol.append(jnp.hstack(SH))
        else: Mcol.append(jnp.eye(2*(2*c.terms+1)))
    M.append(jnp.vstack(Mcol))     
M=jnp.hstack(M)

#M * B_end = B_start, LU decomp
B_start=jnp.hstack(cylinder.full_field)
lu, piv = jax.scipy.linalg.lu_factor(M)
B_end = jax.scipy.linalg.lu_solve((lu, piv), B_start)

def compute_field(time): #struggling to vectorize this?
    #source field   
    VE=jnp.stack([A_E0[order+s_terms]*j1jax(order, k_perp_out * jnp.sqrt(X**2+Y**2))\
                 *jnp.exp(1j*(order * jnp.arctan2(Y,X)-omega * time)) for order in vs_terms])  
    
    output_position=0
    for d in cylinder.instances:
        VEd=jnp.stack([B_end[order+output_position]*h1jax(order, k_perp_out * jnp.sqrt((X-d.pos[0])**2+(Y-d.pos[1])**2))\
                 *jnp.exp(1j*(order * jnp.arctan2(Y-d.pos[1],X-d.pos[0])-omega * time)) for order in d.vterms])
        output_position+=2*(2*d.terms+1) #shift to next set of d
        VE=jnp.concatenate([VE, VEd], axis=0)

    return jnp.real(VE.sum(axis=0))

frames = [jnp.asarray(compute_field(i)) for i in t]
fig, ax = plt.subplots(dpi=200)
c=ax.pcolormesh(X, Y, frames[0], cmap='gist_heat', vmax=color_range,vmin=-color_range)
fig.colorbar(c, ax=ax, label='Scalar Value')
ax.set_title("Plot of scalar field V(r,theta)", va='bottom')

def update(frame):
    c.set_array(frames[frame].flatten())
    return c,
ani = animation.FuncAnimation(
    fig, update, frames=ani_frames, interval=ani_speed, blit=True)
ani.save('field.gif', writer='pillow', fps=10,dpi=200) #brew install ffmpeg #writer for .mp4
