from scipy.optimize import newton



def Euler(U, dt, t, F_E): 

    return (U + dt*F_E(U,t))

def Crank_Nicolson(U, dt, t, F ): 
    def f_CN(X): 
         
         return  X - (U + dt/2*F(U, t)) - dt/2*F(X, t+dt)
 
    return newton( f_CN, U )

def RK4(U, t, dt):

    k1 = Kepler(U, t)
    k2 = Kepler(U + dt*k1/2, t + dt/2)
    k3 = Kepler(U + dt*k2/2, t + dt/2)
    k4 = Kepler(U + dt*k3, t + dt)
    
    return (k1 + 2*k2 + 2*k3 + k4)/6

def Inverse_Euler(U, dt, t, F):
    def f_I(X):
        return X - U - dt*F(X,t)

    return newton(f_I, U)

