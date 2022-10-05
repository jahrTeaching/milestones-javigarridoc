from Functions.Cauchy_Problem import Cauchy_problem
from Functions.Physics import Kepler
from Functions.Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
import matplotlib.pyplot as plt
import numpy as np


N = 10000 # Nº pasos
tf = 10 # valor final
dt = tf/N

# CONDICIONES INICIALES
r0 = np.array([1, 0])
v0 = np.array([0, 1])

U0 = np.hstack((r0,v0)) # Initial condition
print(U0)

t = np.linspace(0, tf, N)
print(t[0])



U_Euler = Cauchy_problem(Euler, Kepler, t, U0)

plt.figure(1)
plt.plot(U_Euler[:,0],U_Euler[:,1])
plt.title('Euler')
plt.show()



U_Crank_Nicolson = Cauchy_problem(Crank_Nicolson, Kepler, t, U0)

plt.figure(2)
plt.plot(U_Crank_Nicolson[:,0],U_Crank_Nicolson[:,1])
plt.title('Crank Nicolson')
plt.show()



U_RK4 = Cauchy_problem(RK4, Kepler, t, U0)

plt.figure(3)
plt.plot(U_RK4[:,0],U_RK4[:,1])
plt.title('Runge-Kutta 4')
plt.show()



U_InverseEuler = Cauchy_problem(Inverse_Euler, Kepler, t, U0)

plt.figure(4)
plt.plot(U_InverseEuler[:,0],U_InverseEuler[:,1])
plt.title('Inverse Euler')
plt.show()
