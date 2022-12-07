from numpy import zeros, reshape, array
from numpy.linalg import norm

G = 6.67e-11 # Nm^2/kg
Nb = 4 # nº cuerpos
Nc = 3 # nº coord
#Nt = (N+1)*2*Nc*Nb

# U --> vector estado
# r,v --> posicion, velocidad 
# dr(i)/dt = v(i)
# dv(i)/dt = sum(j) [G*m(j)*(r(j) - r(i)) / |r(j) - r(i)|^3]

def F_NBody(U, t):
    
    
    U_sol = reshape(U,(Nb,2,Nc)) #Reshape de la U: dividir en arrays para cada cuerpo (i), cada cuerpo tiene en fila 0 la r(i) y en fila 1 la v(i)
    F = zeros(len(U))
    dU_sol = reshape(F, (Nb, 2, Nc)) # Los valores introducidos en dU_sol serán insertados en F

    r = reshape(U_sol[:, 0, :], (Nb, Nc)) # Guarda en array posiciones cada uno de los r(i) --> POSICIONES
    v = reshape(U_sol[:, 1, :], (Nb, Nc)) # Guarda en array velocidades cada uno de los v(i) --> VELOCIDADES
    r_x = array(reshape(r[:, 0], (1, Nb)))
    drdt = reshape(dU_sol[:, 0, :], (Nb, Nc)) # Guarda en array derivada posiciones cada uno de los dr(i)/dt --> VELOCIDADES
    dvdt = reshape(dU_sol[:, 1, :], (Nb, Nc)) # Guarda en array derivada velocidades cada uno de los dv(i)/dt --> ACELERACIONES
    
    #dvdt[:,:] = 0

    for i in range(Nb):
        drdt[i,:] = v[i,:]
        
        for j in range(Nb):
            if j != i:
                dist = r[j,:] - r[i,:]
                dvdt[i,:] = dvdt[i,:] + dist[:]/(norm(dist)**3) # *m_j*G
                
    
    return F

