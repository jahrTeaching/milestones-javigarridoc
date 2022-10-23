from Functions.Cauchy_Problem import Cauchy_problem
from numpy import linspace, zeros, size, log10
from numpy.linalg import norm

def Richardson(Temporal_Scheme, F, t, U0, order):

    N1 = len(t) - 1
    N0 = len(U0)
    Error = zeros((N1+1, N0))
    t1 = t
    t2 = linspace(0, t1[N1-1], 2*size(t1))

    U1 = Cauchy_problem(Temporal_Scheme, F, t1, U0)
    U2 = Cauchy_problem(Temporal_Scheme, F, t2, U0)

    for i in range(N1):
        Error[i,:] = (U2[2*i,:] - U1[i,:]) / (1 - 1/(2**order))

    return Error

def Covergence_Rate(Temporal_Scheme, F, t1, U0):

    N2 = size(t1)
    T = t1[N2-1]
    m = 10
    U1 = Cauchy_problem(Temporal_Scheme, F, t1, U0)

    log_E = zeros(m)
    log_N = zeros(m)

    for i in range(m):

        N2 =  2*N2
        t2 = linspace(0,T,N2)
        U2 = Cauchy_problem(Temporal_Scheme, F, t2, U0)

        log_E[i] = log10(norm(U2[int(N2-1),:] - U1[int(N2/2-1),:]))
        log_N[i] = log10(N2)

        t1 = t2
        U1 = U2


    return [log_E, log_N]