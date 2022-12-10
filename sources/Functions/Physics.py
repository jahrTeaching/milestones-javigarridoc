from numpy import array

def Kepler(U,t):
    
    x = U[0]
    y = U[1]
    dxdt = U[2]
    dydt = U[3]
    d = (x**2 + y**2)**(3/2)

    return array([ dxdt, dydt, -x/d, -y/d])

def Linear_Oscilator(U, t):
    r = U[0]
    drdt = U[1]
    return array([drdt, -r])

