import numpy as np

def Cauchy_problem(Temporal_Scheme, F, t, U0): 

   N = len(t) - 1
   N0 = len(U0)
   U = np.zeros((N+1, N0), dtype=type(U0)) 
   U[0,:] = U0

   for i in range(N):

      U[i+1,:] = Temporal_Scheme(U[i,:], t[i+1] - t[i], t[i], F) 

   return U
