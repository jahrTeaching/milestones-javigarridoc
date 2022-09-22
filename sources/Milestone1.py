import numpy as np
import matplotlib.pyplot as plt

#Euler == U

def Kepler(U,t):
    F0 = U[2]
    F1 = U[3]
    F2 = -U[0]/(U[0]**2 + U[1]**2)**(3/2)
    F3 = -U[1]/(U[0]**2 + U[1]**2)**(3/2)

    return np.array([F0, F1, F2, F3])

def Euler(U, dt, t, F): 

    return (U + dt*F(U, t))

def RK4(U, t, deltat):

    k1 = Kepler(U, t)
    k2 = Kepler(U + deltat*k1/2, t + deltat/2)
    k3 = Kepler(U + deltat*k2/2, t + deltat/2)
    k4 = Kepler(U + deltat*k3, t + deltat)
    
    return (k1 + 2*k2 + 2*k3 + k4)/6

#-------------------------------------#

def F():
    n = 200  #nº de pasos
    tf = 10 #valor final
    dt = tf/n

    # CONDICIONES INICIALES
    r0 = np.array([1, 0])
    v0 = np.array([0, 1])

    U0 = np.hstack((r0,v0)) # Initial condition
    print(U0)


    U = np.zeros((n, 4)) # En cada fila vector r y vector v
    U[0,:] = U0

    t = np.zeros((n, 1))
    t[0] = 0
    F = np.zeros(4)

    for i in range(1,n):
        
        t[i] = dt*i
        F = Kepler(U[i-1,:],t[i])
        print(F)
        
        U[i,:] = U[i-1,:] + dt*F

    #print('Solución Euler')
    #print(U)

    plt.figure(1)
    plt.plot(U[:,0],U[:,1])
    plt.show()


def F_Euler():
    n = 200  #nº de pasos
    tf = 10 #valor final
    dt = tf/n

    # CONDICIONES INICIALES
    r0 = np.array([1, 0])
    v0 = np.array([0, 1])

    U0 = np.hstack((r0,v0)) # Initial condition
    print(U0)


    U = np.zeros((n, 4)) # En cada fila vector r y vector v
    U[0,:] = U0

    t = np.zeros((n, 1))
    t[0] = 0
    F = np.zeros(4)

    for i in range(1,n):
        t[i] = dt*i
        F = Kepler(U[i-1,:],t[i])
        U[i,:] = Euler(U[i-1,:],dt,t[i],F)
        plt.figure(2)
        plt.plot(U[:,0],U[:,1])
        plt.show()


def F_RK4():

    n = 200  #nº de pasos
    tf = 10 #valor final
    dt = tf/n

    # CONDICIONES INICIALES
    r0 = np.array([1, 0])
    v0 = np.array([0, 1])

    U0 = np.hstack((r0,v0)) # Initial condition
    U1 = np.zeros((n,4))
    U1[0,:] = U0

    for i in range(1,n):

        U1[i,:] = U1[i-1,:] + RK4(U1[i-1,:],tf,dt)*dt

    #print('Solución RK4')
    #print(U1)

#--------------------------------------------------#

F()
F_RK4()
