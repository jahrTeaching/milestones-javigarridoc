import numpy as np
import matplotlib.pyplot as plt

n = 10 #nº de pasos
tf = 10 #valor final
dt = tf/n

# CONDICIONES INICIALES
r0 = np.array([1, 0])
v0 = np.array([0, 1])

U0 = np.hstack((r0,v0)) # Initial condition
print(U0)


U = np.zeros((n, 4))
U[0,:] = U0



#Euler
F = np.zeros(4)
print(F)


for i in range(1,n):
    print(i)
    F0 = U[i-1,2]
    F1 = U[i-1,3]
    F2 = -U[i-1,0]/(U[i-1,0]**2 + U[i-1,1]**2)**(3/2)
    F3 = -U[i-1,1]/(U[i-1,0]**2 + U[i-1,1]**2)**(3/2)
    F = np.array([F0,F1,F2,F3])
    print(F)
    U[i,:] = U[i-1,:] + dt*F

print(U)


