import math as mt
import numpy as np
import scipy.optimize as op
from scipy import *

#r est la masse volumique
r=1
#u est la visquosité dynamique
u=1
#notre précision
n=50

#valeurs du système"r
Pe=100
Ps=5
l=20
Dmax=2
C=1
po=0.01
phi = 5

A=8*r/mt.pi
B=128*u/mt.pi

#diamètre en fonction de la pression
def D(x):
    return Dmax*(1/(1+C*(mt.e**(-x/po))))

#notre fonction qui reçoit maintenant un vecteur

def F(p,P,ph):
    return P-p+A*(ph**2)*((1/D(P)**4)-(1/D(p)**4))+(B*ph*(D(p)**(-4))*(l/n))


"""initialisation de la liste des pressions"""
p=[]
for i in range(1,n+1):
    a=(Ps-Pe)/(n-1)
    b=((n*Pe)-Ps)/(n-1)
    p.append(a*i+b)


#initialisation du vecteur des pressions initialisés + valeur initialisée du flux

x0 = np.zeros(shape=(n-1,))

"""for i in range(n-2):
    x0[i] = p[i+1]
x0[n-2]=phi"""



#définir la funct qu'on va implémenter dans la optimise fonction

def Ftot(x):
    t = np.zeros(shape=(n-1,))
    t[0] = F(Pe, x[0], phi)
    for i in range(1,n-2):
        t[i] = F(x[i-1],x[i],phi)
    t[n-2] = F(x[n-3],Ps,phi)
    return t

print(Ftot(x0))

#calling the optimisation function

scipy.optimize.fsolve(Ftot, tuple(x0),xtol=1.49012e-08)






