import torch
from torch.autograd import Function
import scipy.special as sp
torch.set_default_dtype(torch.float32)

# should check to see if scipy implementation of derivative is faster than 0.5 * (sp.jv(n - 1, x) - sp.jv(n + 1, x))

class BesselJv(Function):
    @staticmethod
    def forward(ctx, n, x):
        ctx.save_for_backward(x)
        ctx.n=n
        return sp.jv(n, x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.n
         #pytorch manual requires complex gradient to be returned as conjugate
         #no grad defined for n
        return None, grad_output * torch.conj(0.5 * (sp.jv(n - 1, x) - sp.jv(n + 1, x)))
    
class BesselYv(Function):
    @staticmethod
    def forward(ctx, n, x):
        ctx.save_for_backward(x)
        ctx.n=n
        return sp.yv(n, x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.n
         #pytorch manual requires complex gradient to be returned as conjugate
         #no grad defined for n
        return None, grad_output * torch.conj(0.5 * (sp.yv(n - 1, x) - sp.yv(n + 1, x)))
    
# Wrappers for Torch Bessel functions
def jv(n, x):
    return BesselJv.apply(n, x)
def yv(n, x):
    return BesselYv.apply(n, x)

#Secondary functions
def jvp(n, x):
    return 0.5 * (jv(n - 1, x) - jv(n + 1, x))
def yvp(n, x):
    return 0.5 * (yv(n - 1, x) - yv(n + 1, x))
def h1v(n, x):
    return jv(n, x) + 1j * yv(n, x)
def h1vp(n, x):
    return jvp(n, x) + 1j * yvp(n, x)