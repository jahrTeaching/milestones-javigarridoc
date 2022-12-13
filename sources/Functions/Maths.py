from operator import matmul
from numpy import zeros, size, array, matmul, dot
from numpy.linalg import inv, norm

'''
OPERACIONES MATEMÁTICAS:

  Inputs: 
         U : Vector estado en tn
         dt: Paso de tiempo 
         F : Sistema del que se quiere obtener la matriz Jacobiana o que se quiere solucionar con Newton-Raphson
 
  Outputs: 
 
         J : Matriz jacobiana 
         newton : solución del sistema no linear por Newton-Raphson 

'''


# Jacobian Matrix

def Jacobian(F, U):
	N = size(U)
	J= zeros([N,N])
	t = 1e-10

	for i in range(N):
		xj = zeros(N)
		xj[i] = t
		J[:,i] = (F(U + xj) - F(U - xj))/(2*t)
	return J

# LU Factorization 

def factorization_LU(A):

	N = size(A,1)
	U = zeros([N,N])
	L = zeros([N,N])

	U[0,:] = A[0,:]
	for k in range(0,N):
		L[k,k] = 1

	L[1:N,0] = A[1:N,0]/U[0,0]


	for k in range(1,N):

		for j in range(k,N):
			U[k,j] = A[k,j] - matmul(L[k,0:k], U[0:k,j])

		for i in range(k+1,N):
			L[i,k] =(A[i,k] - matmul(U[0:k,k], L[i,0:k])) / (U[k,k])

	return L@U, L, U

def solve_LU(M,b):

	N=size(b)
	y=zeros(N)
	x=zeros(N)

	A,L,U = factorization_LU(M)
	y[0] = b[0]

	for i in range(0,N):
		y[i] = b[i] - matmul(A[i,0:i], y[0:i])
		

	x[N-1] = y[N-1]/A[N-1,N-1]

	for i in range(N-2,-1,-1):
		x[i] = (y[i] - matmul(A[i, i+1:N+1], x[i+1:N+1])) / A[i,i]
		
	return x

# Inverse Matrix

def Inverse(A):

	N = size(A,1)

	B = zeros([N,N])

	for i in range(0,N):
		one = zeros(N)
		one[i] = 1

		B[:,i] = solve_LU(A, one)

	return B

# Newton-raphson method

def newton(F, U0):
	N = size(U0) 
	U = zeros(N)
	U1 = U0
	error = 1
	stop = 1e-10
	iteration = 0

	while error > stop and iteration < 10000:
		U = U1 - dot(Inverse(Jacobian(F, U1)),F(U1))
		error = norm(U - U1)
		U1 = U
		iteration = iteration +1
	return U