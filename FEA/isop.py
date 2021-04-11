import numpy as np
import sys

# Function Definitions

def constitutiveMat(stressState, E, v):
    planeStress = (E/(1-v**2))*np.array([[1, v, 0], 
                                         [v, 1, 0], 
                                         [0, 0, (1-v)/2]])
    planeStrain = (E/(1+v)/(1-2*v))*np.array([[1-v, v, 0], 
                                              [v, 1-v, 0], 
                                              [0, 0, (1-2*v)/2]])
    axiSym = (E/(1+v)/(1-2*v))*np.array([[1/(1-v), v, v, 0], 
                                         [v, 1/(1-v), v, 0], 
                                         [v, v, 1/(1-v), 0],
                                         [0, 0, 0, (1-2*v)/2]])    
    if stressState == 'PlaneStress':
        return planeStress
    elif stressState == 'PlaneStrain':
        return planeStrain
    elif stressState == 'Axisym':
        return axiSym
    else:
        print('Invalid stress state!')
        sys.exit(1) 

def Jacobi(ksi, eta, elementCoordinates):      
    return 1/4*np.matmul(np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)],
                                   [-(1-ksi), -(1+ksi), (1+ksi), (1-ksi)]]),
                         elementCoordinates)  

def BMatrix(stressState,ksi,eta,elementCoordinates,E, v):
    J = Jacobi(ksi,eta,elementCoordinates)
    T = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
    L = np.zeros([4,4])
    
    L[0:2,0:2] = np.linalg.inv(J)
    L[2:4,2:4] = np.linalg.inv(J)
    Nm = (1/4)*np.array([[-1+eta, 0, 1-eta, 0, 1+eta, 0, -1-eta, 0],
                         [-1+ksi, 0, -1-ksi, 0, 1+ksi, 0, 1-ksi, 0],
                         [0, -1+eta, 0, 1-eta, 0, 1+eta, 0, -1-eta],
                         [0, -1+ksi, 0, -1-ksi, 0, 1+ksi, 0, 1-ksi]])
    B = np.matmul(np.matmul(T,L),Nm)
    if stressState == 'Axisym':
        # Modifications for axisymmetric case, as described in Bathe's 
        # textbook, Finite Element Procedures, Example 5.9. Thickness is
        # calculated for each integration point
        t = np.sum(np.dot((1/4)*np.array([(1-ksi)*(1-eta),
                                          (1+ksi)*(1-eta),(1+ksi)*(1+eta),
                                          (1-ksi)*(1+eta)]),
                          elementCoordinates[:,0]))
        B =  np.vstack([B, 1/4*np.array([(1-ksi)*(1-eta)/t, 0, 
                                         (1+ksi)*(1-eta)/t, 0, 
                                         (1+ksi)*(1+eta)/t,0, 
                                         (1-ksi)*(1+eta)/t, 0])])
    return B

def stiffness(stressState,ksi,eta,elementCoordinates,t,w1,w2,E, v):
    J = Jacobi(ksi,eta,elementCoordinates)
    C = constitutiveMat(stressState, E, v)
    B = BMatrix(stressState,ksi,eta,elementCoordinates,E, v)
    if stressState == 'Axisym':
        t = np.sum(np.dot((1/4)*np.array([(1-ksi)*(1-eta),(1+ksi)*(1-eta),
                                          (1+ksi)*(1+eta),(1-ksi)*(1+eta)]),
                          elementCoordinates[:,0]))
    if stressState == 'PlaneStrain':
        t = 1    
    return w1*w2*t*np.linalg.det(J)*np.matmul(np.matmul(np.transpose(B),C),B)

# 3x3 Gaussian quadrature integration points    
points = np.array([[-np.sqrt(0.6),-np.sqrt(0.6)],[-np.sqrt(0.6),0],
                   [-np.sqrt(0.6),np.sqrt(0.6)],[0,-np.sqrt(0.6)],[0,0],
                   [0,np.sqrt(0.6)],[np.sqrt(0.6),-np.sqrt(0.6)],
                   [np.sqrt(0.6),0],[np.sqrt(0.6),np.sqrt(0.6)]])
# Weights
weights = np.array([[5/9,5/9], [5/9,8/9],[5/9,5/9],[8/9,5/9],[8/9,8/9],
                    [8/9,5/9],[5/9,5/9],[5/9,8/9],[5/9,5/9]])

# User Input
elementCoordinates = np.array([[0, 0],[20,0],[20,20],[0,20]])
t = 20 # Thickness of the element
E = 200000 # Modulus of elasticity
v = 0.30 # Poisson's ratio
stressState = 'PlaneStress'# Valid inputs: 'PlaneStrain','PlaneStress','Axisym'

K = np.zeros([8,8])
for integrationPoint in range(9):
    ksi = points[integrationPoint,0]
    eta = points[integrationPoint,1]
    w1 = weights[integrationPoint,0]
    w2 = weights[integrationPoint,1]
    K = K + stiffness(stressState,ksi,eta,elementCoordinates,t,w1,w2,E, v)
