# import jax.numpy as np
import numpy as np

def ge_dist(Lpinv, x1, x2):
    dx = (x2 - x1)
    dm = np.einsum('...i,...i->...',dx.dot(Lpinv),dx)
    return np.sqrt(dm)

# @jit
def laplacian(a):
    d = np.diag(a.sum(axis=0))
    return d - a


def symlaplacian(a):
    dinv2 = np.diag(1./np.sqrt(a.sum(0)))
    return dinv2@laplacian(a)@dinv2


def rwlaplacian(a):
    dinv = np.diag(1./a.sum(0))
    return dinv@laplacian(a)


def lin_interp_array(X, T, t):
    assert (T[0] <= t) and (t <= T[-1]), "requested time is not within bounds of `T`!"
    
    idx0 = np.searchsorted(T, t)
    idx1 = idx0 + 1    
    slope = (X[idx1,...] - X[idx0,...])/(T[idx1]-T[idx0])
    
    return np.clip(X[idx0,...] + (t - T[idx0])*slope, a_min=0.)

# @jit
def df_dt(f, t, L, k):
    return k*np.matmul(-L,f[...,None]).squeeze(-1)
#     return -k*L.dot(f)
#
# def diffusion(a, t):
    
    
def cumulative_diffusion(a, t):
    kern = diffusion(a, t)
    return np.linalg.pinv(a)@(kern - np.eye(a.shape[0]))

def _cumulative_heat_ode(L, t, dt=None):
    time_hist = _heat_ode(L, t, dt=dt)
    riemann_sum = np.cumsum(time_hist*dt, axis=0)
    return riemann_sum[-1]

def cumulative_heat_ode(a, t, dt=None):
    return _cumulative_heat_ode(laplacian(a), t, dt=dt)[-1]

def cumulative_norm_heat_ode(a, t, dt=None):
    return _cumulative_heat_ode(symlaplacian(a), t, dt=dt)[-1]

def cumulative_pagerank_heat_ode(a, t, dt=None):
    return _cumulative_heat_ode(rwlaplacian(a), t, dt=dt)[-1]

def cumulative_diffusion_ode(a, t, dt=None):
    time_hist = _diffusion_ode(a, t, dt=dt)
    riemann_sum = np.cumsum(time_hist*dt, axis=0)
    return riemann_sum[-1]
