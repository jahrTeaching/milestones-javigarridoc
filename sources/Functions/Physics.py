from numpy import array, sqrt, zeros
from numpy.linalg import eig
from scipy.optimize import fsolve
from Functions.Maths import Jacobian, newton

def Kepler(U,t):
    
    x = U[0]
    y = U[1]
    dxdt = U[2]
    dydt = U[3]
    d = (x**2 + y**2)**(3/2)

    return array([dxdt, dydt, -x/d, -y/d])

def Linear_Oscilator(U, t):
    r = U[0]
    drdt = U[1]
    return array([drdt, -r])

def CR3BP(U,t,mu):
    r_x, r_y = U[0], U[1] # Posicion
    v_x, v_y = U[2], U[3] # Velocidad
    
    r1 = sqrt((r_x + mu)**2 + r_y**2)
    r2 = sqrt((r_x - 1 + mu)**2 + r_y**2)

    dvdt_x = 2*v_y + r_x - ((1 - mu)*(r_x + mu))/(r1**3) - mu*(r_x - 1 + mu)/(r2**3)
    dvdt_y = -2*v_x + r_y -((1 - mu)/(r1**3) + mu/(r2**3))*r_y

    return array([v_x, v_y, dvdt_x, dvdt_y])

def Lagrange_Points_Calculation(U0, NL, mu):

    LP = zeros([5,2])

    def F(Y):
        
        X = zeros(4)
        X[0:2] = Y
        X[2:4] = 0
        FX = CR3BP(X, 0, mu)
        return FX[2:4]
        
    for i in range(NL):
        LP[i,:] = fsolve(F, U0[i,0:2])

    return LP

def LP_Stability(U0, mu):

    def F(Y):
        return CR3BP(Y, 0 , mu)

    A = Jacobian(F, U0)
    values, vectors = eig(A)

    return values