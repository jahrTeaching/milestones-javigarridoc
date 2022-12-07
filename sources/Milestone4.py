from Functions.Physics import Kepler, Linear_Oscilator
from Functions.Temporal_Schemes import Euler, Crank_Nicolson, RK4, Inverse_Euler, LeapFrog
from Functions.Cauchy_Problem import Cauchy_problem
from Functions.Stability import Stability_Region
import matplotlib.pyplot as plt
from numpy import array, linspace, zeros, size


tf = 20 # valor final
dt = [0.1, 0.01, 0.001]

# CONDICIONES INICIALES
U0 = array([1,0])
print(U0)


Temp_Schemes = [Euler, RK4, Crank_Nicolson, Inverse_Euler, LeapFrog]
T_S_list = ['Euler', 'RK4', 'CrankNicolson', 'InverseEuler', 'LeapFrog']
colors = ['b--','r--','g--']
print(size(Temp_Schemes))

# INTEGRAR OSCILADOR LINEAR
for i in range(size(Temp_Schemes)):


    for j in range (size(dt)):
        N = int(tf/dt[j])
        t = linspace(0, tf, N)
        U = Cauchy_problem( Temp_Schemes[i], Linear_Oscilator, t, U0)

        plt.figure(i)
        plt.plot(t, U[:,0],colors[j], label = 'dt =' + str(dt[j]) + ' s')
        plt.title(f'Oscilador integrado con {T_S_list[i]}')
        plt.xlabel("t (s)")
        plt.ylabel("x")
        plt.legend(loc = "lower left")
        plt.grid()
    #plt.show()


# REGIÓN DE ESTABILIDAD
a = linspace(-5, 5, num=100);
b = linspace(-5, 5, num=100);


for i in range(size(Temp_Schemes)):
    Stab_Region = Stability_Region(Temp_Schemes[i],a,b)
    print(Stab_Region)
    plt.figure(i+size(Temp_Schemes))

    if Temp_Schemes[i] == 'LeapFrog':
        Im = linspace(-1,1,100)
        Re = zeros(100)
        print('im',Im)
        print('re',Re)
        plt.plot(Re, Im, color = '#b300ff')
        plt.show()
        
    else:
        plt.contour(a,b, Stab_Region, levels = [0, 1], colors = ['#b300ff'])
        plt.contourf(a,b, Stab_Region, levels = [0, 1], colors =['#78745d'])
    
    colors = ['b','r','g']
    
    for j in range(len(dt)):
        plt.plot([0,0], [dt[j],-dt[j]], 'o', color = colors[j], label = 'dt =' + str(dt[j]) + ' s')
    plt.ylabel("Im")
    plt.xlabel("Re")
    plt.title(f'Región de estabilidad de {T_S_list[i]}')
    plt.legend()
    plt.grid()
    plt.show()




