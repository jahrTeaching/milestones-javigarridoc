from Functions.Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler, LeapFrog
from Functions.Cauchy_Problem import Cauchy_problem
from Functions.N_Body import F_NBody
from Functions.Physics import Kepler, CR3BP, Lagrange_Points_Calculation, LP_Stability
from Functions.EmbRK import Embedded_RK

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import array, linspace, zeros, size, linspace, reshape, around
from numpy.random import random

# Time definition
N = int(1e6) # time steps
t0 = 0 # timepo inicial
tf = 500 # tiempo final
#dt = [0.1, 0.01, 0.001]
#N = int(tf/dt[j])
t = linspace(t0, tf, N)

### CR3BP SOLUTION ###

#mu = 3.0039e-7  # Earth - Sun
mu = 1.2151e-2 # Earth - Moon

def F(U, t):
    return CR3BP(U, t, mu)

### LAGRANGE POINTS ###

NLP = 5 # Number of Lagrange Points

U0 = zeros([NLP,4])  # Initial values
U0[0,:] = array([0.8, 0.6, 0, 0])
U0[1,:] = array([0.8, -0.6, 0, 0])
U0[2,:] = array([-0.1, 0, 0, 0])
U0[3,:] = array([0.1, 0, 0, 0])
U0[4,:] = array([1.01, 0, 0, 0])

LagrangePoints = Lagrange_Points_Calculation(U0, NLP, mu)
print(LagrangePoints)
### ORBITS AROUND LP ###
U0LP = zeros(4)
U0SLP = zeros(4)

eps = 1e-3*random()
print('Choose Lagrange Point: 1, 2, 3, 4, 5')
sel_LP_ = int(input())  # Selected Lagrange Point
if sel_LP_ == 1:
    sel_LP = 4
elif sel_LP_ == 2:
    sel_LP = 5
elif sel_LP_ == 3:
    sel_LP = 3
elif sel_LP_ == 4:
    sel_LP = 1
elif sel_LP_ == 5:
    sel_LP = 2



U0LP[0:2] = LagrangePoints[sel_LP-1,:] + eps
U0LP[2:4] = eps

U0SLP[0:2] = LagrangePoints[sel_LP-1,:] 
U0SLP[2:4] = 0

Temp_Schemes = [Euler, RK4, Crank_Nicolson, Inverse_Euler, LeapFrog, Embedded_RK]
T_S_list = ['Euler', 'RK4', 'CrankNicolson', 'InverseEuler', 'LeapFrog', 'Embedded_RK']
colors = ['b','r','g','m','y','c']

### STABILITY OF LAGRANGE POINTS ###
print('Choose temporal scheme: Euler[0], RK4[1], Crank_Nicolson[2], Inverse_Euler[3], LeapFrog[4], Embedded_RK[5]')
k = int(input())
TS = Temp_Schemes[k]
T_S = T_S_list[k]
U_LP = Cauchy_problem(TS, F, t, U0LP)
eig = LP_Stability(U0SLP, mu)
print(around(eig.real,8))

### PLOT ###
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(U_LP[:,0], U_LP[:,1],'-',color = "r")
ax1.plot(-mu, 0, 'o', color = "g")
ax1.plot(1-mu, 0, 'o', color = "b")

for i in range(NLP):
    ax1.plot(LagrangePoints[i,0], LagrangePoints[i,1] , 'o', color = "k")

ax1.set_xlim(-9,9)
ax1.set_ylim(-9,9)
ax1.set_title("Orbital system view")

ax2.plot(U_LP[:,0], U_LP[:,1],'-',color = "r")
ax2.plot(LagrangePoints[sel_LP-1,0], LagrangePoints[sel_LP-1,1] , 'o', color = "k")


ax2.set_title("Lagrange point view")
ax2.set_xlim(LagrangePoints[sel_LP-1,0]-0.25,LagrangePoints[sel_LP-1,0]+0.25)
ax2.set_ylim(LagrangePoints[sel_LP-1,1]-0.25,LagrangePoints[sel_LP-1,1]+0.25)
fig.suptitle(f"Earth-Sun - CR3BP ({T_S} TS) - Orbit around the L{sel_LP_} point with t = " + str(t[N-1])+'s')

for ax in fig.get_axes():
    ax.set(xlabel='x', ylabel='y')
    ax.grid()

plt.show()