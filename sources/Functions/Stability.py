from Functions.Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler, LeapFrog
from numpy import array, zeros, size, sqrt, float64, absolute


def Stability_Region(Temporal_Scheme,x,y):
    n = size(x)
    W = zeros([n,n],dtype=complex)

    for i in range(n):
        for j in range(n):
            W[n-1-j,i] = complex(x[i],y[j])

    return absolute(array(Stability_Polynomial(Temporal_Scheme, W)))

def Stability_Region2(Temporal_Scheme,x,y):
    n = size(x)
    W = zeros([n,n],dtype=float64)

    for i in range(n):
        for j in range(n):
            Z = complex(x[i], y[j])
            r = Temporal_Scheme(1.,1.,lambda u, t: Z*u, 0.)

            W[i,j] = abs(r)

    return W


def Stability_Polynomial(Temporal_Scheme, w):
    if Temporal_Scheme == Euler:
        r = 1 + w
    elif Temporal_Scheme == Inverse_Euler:
        r = 1/(1-w)
    elif Temporal_Scheme == Crank_Nicolson:
        r = (2+w)/(2-w)
    elif Temporal_Scheme == RK4:
        r = 1 + w + (w**2)/2 + (w**3)/6 + (w**4)/24
    elif Temporal_Scheme == LeapFrog:
        r = sqrt(1)

    return r