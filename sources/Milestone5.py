from Functions.Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler, LeapFrog
from Functions.Cauchy_Problem import Cauchy_problem
from Functions.N_Body import F_NBody
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import array, linspace, zeros, size, linspace, reshape

N = 10 # time steps
t0 = 0 # timepo inicial
tf = 5 # tiempo final
dt = [0.1, 0.01, 0.001]
#N = int(tf/dt[j])
t = linspace(t0, tf, N)

Nb = 4 # nº cuerpos
Nc = 3 # nº coord

colors = ['b','r','g','m','y','c']
# CONDICIONES INICIALES

U0 = zeros(Nb*2*Nc)
U_0 = reshape(U0, (Nb, 2, Nc) )
r0 = reshape(U_0[:, 0, :], (Nb, Nc))
v0 = reshape(U_0[:, 1, :], (Nb, Nc))

# body 1 
r0[0,:] = [ 2, 2, 0]
v0[0,:] = [ -0.4, 0, 0]

# body 2 
r0[1,:] = [ -2, 2, 0] 
v0[1,:] = [ 0, -0.4, 0]

# body 3 
r0[2, :] = [ -2, -2, 0 ] 
v0[2, :] = [ 0.4, 0., 0. ] 
         
# body 4 
r0[3, :] = [ 2, -2, 0 ] 
v0[3, :] = [ 0., 0.4, 0. ]

# print(U0)
# print(U_0)
# SOLUCIÓN

U = Cauchy_problem(RK4, F_NBody, t, U0)

#print(size(U))


U_s  = reshape( U, (N, Nb, 2, Nc) ) 
r   = reshape( U_s[:, :, 0, :], (N, Nb, Nc) )

#print('Us = ',U_s)
print('r = ',r[:, 0, 0])

# 2D PLOT
for i in range(Nb):
    plt.figure(1)
    plt.plot(r[:, i, 0], r[:, i, 1], colors[i])
plt.title(f'N = {Nb} body problem: x - y projection')
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.xlim((-4,6))
plt.ylim((-6,4))
plt.grid(True)
plt.show()

'''
for i in range(Nb):
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(r[:, i, 0], r[:, i, 1], r[:, i, 2], colors[i])
plt.title(f'N = {Nb} body problem: 3D projection')
plt.xlabel("x")
plt.ylabel("y")
#plt.axis('equal')
#plt.xlim((-2,2))
#plt.ylim((-2,2))
#plt.grid(True)
plt.show()
'''

# 3D PLOT
fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
ax.plot(r[:, 0, 0], r[:, 0, 1], r[:, 0, 2], colors[0])
ax.plot(r[:, 1, 0], r[:, 1, 1], r[:, 1, 2], colors[1])
ax.plot(r[:, 2, 0], r[:, 2, 1], r[:, 2, 2], colors[2])
ax.plot(r[:, 3, 0], r[:, 3, 1], r[:, 3, 2], colors[3])
plt.title(f'N = {Nb} body problem: 3D projection')
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
ax.set_xlim3d(-10, 20) # viewrange for z-axis should be [-4,4] 
ax.set_ylim3d(-20,5) # viewrange for y-axis should be [-2,2] 
ax.set_zlim3d(-5, 2) # viewrange for x-axis should be [-2,2] 
#plt.grid(True)
plt.show()
