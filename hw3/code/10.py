from math import exp

def E_uv(u, v):
    return exp(u) + exp(2*v) + exp(u*v) + u**2 - 2*u*v + 2*(v**2) - 3*u - 2*v

def grad_uv(u, v):
    grad_u = exp(u) + v*exp( u*(v-1) )*exp(u) + 2*u - 2*v - 3
    grad_v = 2*exp(v)*exp(v) + u*exp( v*(u-1) )*exp(v) - 2*u + 4*v - 2
    return grad_u, grad_v

def grad_grad_uv(u, v):
    grad_u = exp(u) + v*exp( u*(v-1) )*exp(u) + 2*u - 2*v - 3
    grad_v = 2*exp(v)*exp(v) + u*exp( v*(u-1) )*exp(v) - 2*u + 4*v - 2
    return grad_u, grad_v

def update_uv(u, v, eta):
    grad_u, grad_v = grad_uv(u, v)
    u -= eta * grad_u
    v -= eta * grad_v
    return u, v

u = 0
v = 0
eta = 1

for t in range(6):
    print(t, ':', u, v, E_uv(u, v))
    u, v = update_uv(u, v, eta)
