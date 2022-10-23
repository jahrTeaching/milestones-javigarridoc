#from Functions.Cauchy_Problem import Cauchy_problem
from Functions.Physics import Kepler
from Functions.Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler
from Functions.Error import Richardson, Covergence_Rate
import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, size, hstack

N = 20000 # Nº pasos
tf = 20# valor final
dt = tf/N
print(dt)

# CONDICIONES INICIALES
r0 = array([1, 0])
v0 = array([0, 1])

U0 = hstack((r0,v0)) # Initial condition

t = linspace(0, tf, N)

# CALCULOS

Error_Euler = Richardson(Euler, Kepler, t, U0, 1)
Norm_Euler = zeros(size(Error_Euler[:,0]))
for i in range(0,size(Error_Euler[:,0])):
     Norm_Euler[i] = (Error_Euler[i,0]**2 + Error_Euler[i,1]**2)**(1/2)

Error_InvEuler = Richardson(Inverse_Euler, Kepler, t, U0, 1)
Norm_InvEuler = zeros(size(Error_InvEuler[:,0]))
for i in range(0,size(Error_InvEuler[:,0])):
     Norm_InvEuler[i] = (Error_InvEuler[i,0]**2 + Error_InvEuler[i,1]**2)**(1/2)

Error_CN = Richardson(Crank_Nicolson, Kepler, t, U0, 2)
Norm_CN = zeros(size(Error_CN[:,0]))
for i in range(0,size(Error_CN[:,0])):
     Norm_CN[i] = (Error_CN[i,0]**2 + Error_CN[i,1]**2)**(1/2)

Error_RK4 = Richardson(RK4, Kepler, t, U0, 4)
Norm_RK4 = zeros(size(Error_RK4[:,0]))
for i in range(0,size(Error_RK4 [:,0])):
     Norm_RK4[i] = (Error_RK4 [i,0]**2 + Error_RK4 [i,1]**2)**(1/2)


[log_E_Euler, log_N_Euler] = Covergence_Rate(Euler, Kepler, t, U0)
[log_E_InvEuler, log_N_InvEuler] = Covergence_Rate(Inverse_Euler, Kepler, t, U0)
[log_E_CN, log_N_CN] = Covergence_Rate(Crank_Nicolson, Kepler, t, U0)
[log_E_RK4, log_N_RK4] = Covergence_Rate(RK4, Kepler, t, U0)


# PLOTS

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t, Error_Euler[:,0],"r", label = "X")
plt.plot(t, Error_Euler[:,1],"b", label = "Y")
plt.title(f'Error de Euler con dt = {dt} s y T = {tf} s')
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "lower left")
plt.subplot(2, 1, 2)
plt.plot(t, Norm_Euler, "g", label = "Module")
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "upper left")
plt.show()

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(t, Error_InvEuler[:,0],"r", label = "X")
plt.plot(t, Error_InvEuler[:,1],"b", label = "Y")
plt.title(f'Error de Euler Inverso con dt = {dt} s y T = {tf} s')
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "lower left")
plt.subplot(2, 1, 2)
plt.plot(t, Norm_InvEuler, "g", label = "Module")
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "upper left")
plt.show()

plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(t, Error_CN[:,0],"r", label = "X")
plt.plot(t, Error_CN[:,1],"b", label = "Y")
plt.title(f'Error de Crank Nicolson con dt = {dt} s y T = {tf} s')
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "lower left")
plt.subplot(2, 1, 2)
plt.plot(t, Norm_CN, "g", label = "Module")
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "upper left")
plt.show()

plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(t, Error_RK4[:,0],"r", label = "X")
plt.plot(t, Error_RK4[:,1],"b", label = "Y")
plt.title(f'Error de Runge-Kutta 4 con dt = {dt} s y T = {tf} s')
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "lower left")
plt.subplot(2, 1, 2)
plt.plot(t, Norm_RK4, "g", label = "Module")
plt.grid(True)
plt.xlabel("t (s)")
plt.ylabel("Error")
plt.legend(loc = "upper left")
plt.show()


plt.figure(5)
plt.plot(log_N_Euler, log_E_Euler, "b", label = "Euler")
plt.title('Euler')
plt.grid(True)
plt.xlabel("log(N)")
plt.ylabel("log(U2-U1)")
plt.legend(loc ='lower left')
plt.show()


plt.figure(6)
plt.plot(log_N_InvEuler, log_E_InvEuler, "b", label = "Inverse Euler")
plt.title('Inverse Euler')
plt.grid(True)
plt.xlabel("log(N)")
plt.ylabel("log(U2-U1)")
plt.legend(loc ='lower left')
plt.show()


plt.figure(7)
plt.plot(log_N_CN, log_E_CN, "b", label = "Crank Nicolson")
plt.title('Crank Nicolson')
plt.grid(True)
plt.xlabel("log(N)")
plt.ylabel("log(U2-U1)")
plt.legend(loc ='lower left')
plt.show()


plt.figure(8)
plt.plot(log_N_RK4, log_E_RK4, "b", label = "Runge-Kutta 4")
plt.title('Runge-Kutta 4')
plt.grid(True)
plt.xlabel("log(N)")
plt.ylabel("log(U2-U1)")
plt.legend(loc ='lower left')
plt.show()