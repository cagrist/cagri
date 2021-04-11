import numpy as np

# Function Definitions

def constitutiveMat(E, v):
    return (E/(1+v))*np.array([[(1-v)/(1-2*v), v/(1-2*v),v/(1-2*v),0,0,0],
                                  [v/(1-2*v),(1-v)/(1-2*v),v/(1-2*v),0,0,0],
                                  [v/(1-2*v),v/(1-2*v),(1-v)/(1-2*v),0,0,0],
                                  [0,0,0,1/2,0,0],
                                  [0,0,0,0,1/2,0],
                                  [0,0,0,0,0,1/2]])

def Jacobi(ksi, eta, zeta, elementCoordinates):      
    return 1/8*np.matmul(np.array([[-(1-eta)*(1+zeta),-(1-eta)*(1-zeta),-(1+eta)*(1-zeta),-(1+eta)*(1+zeta),(1-eta)*(1+zeta),(1-eta)*(1-zeta),(1+eta)*(1-zeta),(1+eta)*(1+zeta)],
                                    [-(1-ksi)*(1+zeta),-(1-ksi)*(1-zeta),(1-ksi)*(1-zeta),(1-ksi)*(1+zeta),-(1+ksi)*(1+zeta),-(1+ksi)*(1-zeta),(1+ksi)*(1-zeta),(1+ksi)*(1+zeta)],
                                    [(1-ksi)*(1-eta),-(1-ksi)*(1-eta),-(1-ksi)*(1+eta),(1-ksi)*(1+eta),(1+ksi)*(1-eta),-(1+ksi)*(1-eta),-(1+ksi)*(1+eta),(1+ksi)*(1+eta)]])
                         ,elementCoordinates)  

def BMatrix(ksi,eta,zeta,elementCoordinates,E, v):
    J = Jacobi(ksi,eta,zeta,elementCoordinates)
    T = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 1, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0, 0]])
    L = np.zeros([9,9])
    L[0:3,0:3] = np.linalg.inv(J)
    L[3:6,3:6] = np.linalg.inv(J)
    L[6:,6:] = np.linalg.inv(J)
    Nm = (1/8)*np.array([[-(1-eta)*(1+zeta),0,0,-(1-eta)*(1-zeta),0,0,-(1+eta)*(1-zeta),0,0,-(1+eta)*(1+zeta),0,0,(1-eta)*(1+zeta),0,0,(1-eta)*(1-zeta),0,0,(1+eta)*(1-zeta),0,0,(1+eta)*(1+zeta),0,0],
                                    [-(1-ksi)*(1+zeta),0,0,-(1-ksi)*(1-zeta),0,0,(1-ksi)*(1-zeta),0,0,(1-ksi)*(1+zeta),0,0,-(1+ksi)*(1+zeta),0,0,-(1+ksi)*(1-zeta),0,0,(1+ksi)*(1-zeta),0,0,(1+ksi)*(1+zeta),0,0],
                                    [(1-ksi)*(1-eta),0,0,-(1-ksi)*(1-eta),0,0,-(1-ksi)*(1+eta),0,0,(1-ksi)*(1+eta),0,0,(1+ksi)*(1-eta),0,0,-(1+ksi)*(1-eta),0,0,-(1+ksi)*(1+eta),0,0,(1+ksi)*(1+eta),0,0],                                   
                                    [0,-(1-eta)*(1+zeta),0,0,-(1-eta)*(1-zeta),0,0,-(1+eta)*(1-zeta),0,0,-(1+eta)*(1+zeta),0,0,(1-eta)*(1+zeta),0,0,(1-eta)*(1-zeta),0,0,(1+eta)*(1-zeta),0,0,(1+eta)*(1+zeta),0],
                                    [0,-(1-ksi)*(1+zeta),0,0,-(1-ksi)*(1-zeta),0,0,(1-ksi)*(1-zeta),0,0,(1-ksi)*(1+zeta),0,0,-(1+ksi)*(1+zeta),0,0,-(1+ksi)*(1-zeta),0,0,(1+ksi)*(1-zeta),0,0,(1+ksi)*(1+zeta),0],
                                    [0,(1-ksi)*(1-eta),0,0,-(1-ksi)*(1-eta),0,0,-(1-ksi)*(1+eta),0,0,(1-ksi)*(1+eta),0,0,(1+ksi)*(1-eta),0,0,-(1+ksi)*(1-eta),0,0,-(1+ksi)*(1+eta),0,0,(1+ksi)*(1+eta),0],                                   
                                    [0,0,-(1-eta)*(1+zeta),0,0,-(1-eta)*(1-zeta),0,0,-(1+eta)*(1-zeta),0,0,-(1+eta)*(1+zeta),0,0,(1-eta)*(1+zeta),0,0,(1-eta)*(1-zeta),0,0,(1+eta)*(1-zeta),0,0,(1+eta)*(1+zeta)],
                                    [0,0,-(1-ksi)*(1+zeta),0,0,-(1-ksi)*(1-zeta),0,0,(1-ksi)*(1-zeta),0,0,(1-ksi)*(1+zeta),0,0,-(1+ksi)*(1+zeta),0,0,-(1+ksi)*(1-zeta),0,0,(1+ksi)*(1-zeta),0,0,(1+ksi)*(1+zeta)],
                                    [0,0,(1-ksi)*(1-eta),0,0,-(1-ksi)*(1-eta),0,0,-(1-ksi)*(1+eta),0,0,(1-ksi)*(1+eta),0,0,(1+ksi)*(1-eta),0,0,-(1+ksi)*(1-eta),0,0,-(1+ksi)*(1+eta),0,0,(1+ksi)*(1+eta)]])
    
    B = np.matmul(np.matmul(T,L),Nm)
    return B

def stiffness(ksi,eta,zeta,elementCoordinates,w1,w2,w3,E, v):
    J = Jacobi(ksi,eta,zeta,elementCoordinates)
    C = constitutiveMat(E, v)
    B = BMatrix(ksi,eta,zeta,elementCoordinates,E, v)
    return w1*w2*w3*np.linalg.det(J)*np.matmul(np.matmul(np.transpose(B),C),B) 


# 3x3x3 Gaussian quadrature integration points
gauss = np.array([-np.sqrt(0.6),0,np.sqrt(0.6)])
weight = np.array([5/9,8/9,5/9])
points = np.zeros([27,3])
weights = np.zeros([27,3])

gaussPoint = 0
for i in range(len(gauss)):
    for j in range(len(gauss)):
        for k in range(len(gauss)):
            points[gaussPoint,:] = np.array([gauss[i],gauss[j],gauss[k]]) 
            weights[gaussPoint,:] = np.array([weight[i],weight[j],weight[k]])
            gaussPoint = gaussPoint+1
                
# User Input
elementCoordinates = np.array([[0,0,0],[0,0,-20],[0,20,-20],[0,20,0],[20,0,0],[20,0,-20],[20,20,-20],[20,20,0]])
E = 200000 # Modulus of elasticity, N/mm2
v = 0.30 # Poisson's ratio

K = np.zeros([24,24])
for integrationPoint in range(gaussPoint):
    ksi = points[integrationPoint,0]
    eta = points[integrationPoint,1]
    zeta = points[integrationPoint,2]
    w1 = weights[integrationPoint,0]
    w2 = weights[integrationPoint,1]
    w3 = weights[integrationPoint,2]
    Bmat = BMatrix(ksi,eta,zeta,elementCoordinates,E, v)
    Jac = Jacobi(ksi, eta, zeta, elementCoordinates)
    K = K + stiffness(ksi,eta,zeta,elementCoordinates,w1,w2,w3,E, v)
