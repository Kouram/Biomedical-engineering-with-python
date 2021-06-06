# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:04:56 2019

@author: kouram
"""

import math as mt
import numpy as np
import matplotlib.pyplot as plt
#r est la masse volumique
r=1
#u est la visquosité dynamique
u=1
#notre précision
n=50
#valeurs du système"r
Pe=52000
Ps=10000
l=20
Dmax=2
C=1
po= 100
phi = 5


A=8*r/mt.pi
B=128*u/mt.pi

#diamètre en fonction de la pression
def D(x):
    return Dmax*(1/(1+C*(mt.e**(-x/po))))

def Ddev(x):
    return C*(Dmax/po)*(mt.e**(-x/po))/((1+C*(mt.e**(-x/po)))**2)

#phi=(Ps-P2)*mt.pi*((D(Ps))**4)/(128*u*l)
#print(phi)

def F(p,P,ph):
    return P-p+A*(ph**2)*((1/D(P)**4)-(1/D(p)**4))+(B*ph*(D(p)**(-4))*(l/n))

def Fi_Piplus(p,P,ph):
    return 1 - 32*r*(ph**2)*Ddev(P)/(mt.pi*D(P)**5)

def Fi_ph(p,P,ph):
    return (16*r/mt.pi*D(P)**4)*ph - 16*r*ph/((mt.pi**2)*D(p)**4) + (128*u/(mt.pi*(D(p)**4)))*(l/n)

def Fiplus_Piplus(p,P,ph):
    return -1 + 32*r*((ph**2)/mt.pi*(D(P)**5))*Ddev(P) - ((512*u*ph*Ddev(P))/(mt.pi*(D(P)**5)))*(l/n)

#def Fiplus_ph(p,P,ph):
    #return (16*r*ph)/(mt.pi*(D(Ps)**4)) - (16*r*ph)/((mt.pi**2)*(D(P)**4)) + (128*u)/(mt.pi*(D(P)**4))

"""initialisation de la liste des pressions"""
p=[]
for i in range(1,n+1):
    a=(Ps-Pe)/(n-1)
    b=((n*Pe)-Ps)/(n-1)
    p.append(a*i+b)

"""initialisation du vecteur colonne des fonctions F"""

Farray = np.zeros(shape=(n-1,1))
for i in range(n-1):
    Farray[i][0] = F(p[i],p[i+1],phi)

"""initialisation de la matrice jacobienne"""
J = np.zeros(shape=(n-1,n-1))
for i in range(n-1):
    for j in range(n-1):
        if j==n-2:
            J[i][j]= Fi_ph(p[i],p[i+1],phi)
        else:
            if i==j:
                J[i][j]= Fi_Piplus(p[i],p[i+1],phi)
            else:
                if i==j+1:
                    J[i][j]= Fiplus_Piplus(p[i],p[i+1],phi)


"""calcul du premier vecteur colonne des erreurs"""
x = np.linalg.solve(J, Farray)
print(x)

"""implémentation de la boucle"""

"""initialisation du vecteur colonne contenant les pi de 2 à n-1 et le flux"""
y = np.zeros(shape=(n-1,1))

for i in range(n-2):
    y[i][0] = p[i+1]
y[n-2] = phi


"""norme de vecteur colonne Farray"""
def normeF(F):
    s=0
    for i in range(n-1):
        s = s + (Farray[i][0])**2
    return mt.sqrt(s)

while np.linalg.norm(Farray) > 0.01 :
#for i in range(1000):
    y = y + x

    """modification des valeurs de pression et de flux"""
    for i in range(1,n-1):
        p[i] = y[i-1]
    phi = y[n-2]
    """initialisation de la matrice jacobienne"""
    J = np.zeros(shape=(n-1,n-1))
    for i in range(n-1):
        for j in range(n-1):
            if j==n-2:
                J[i][j]= Fi_ph(p[i],p[i+1],phi)
            else:
                if i==j and j!=n-2:
                    J[i][j]= Fi_Piplus(p[i],p[i+1],phi)
                else:
                    if i==j+1:
                        J[i][j]= Fiplus_Piplus(p[i],p[i+1],phi)


    Farray = np.zeros(shape=(n-1,1))
    for i in range(n-1):
        Farray[i][0] = -F(p[i],p[i+1],phi)

    x = np.linalg.solve(J, Farray)
print("x = " , x)
print("y = " , y)

print(np.linalg.norm(Farray))


"""traçer la courbe P en fonction de x"""

# Prepare the data
x = np.linspace(0, l, n)

"""liste des pressions finales"""
def pressions(y):
    f = []
    f.append(Pe)
    for i in range(1,n-1):
        f.append(y[i-1][0])
    f.append(Ps)
    return f

def Diamètres(f):
    h = []
    for i in range(len(f)):
        h.append(D(f[i]))
    return h

def rayons(f):
    h=[]
    for i in range(len(f)):
        h.append(0.5*D(f[i]))
    return h


# Plot the data
plt.plot(x, pressions(y))
plt.xlabel('x')
plt.ylabel('Pression')

# Show the plot
plt.show()

"""traçer R en fonction de p"""
# Plot the data
plt.plot(pressions(y),rayons(pressions(y)))
plt.xlabel('pression')
plt.ylabel('rayon')

# Show the plot
plt.show()

"""traçer D en fonction de x"""
# Plot the data
plt.plot(x,rayons(pressions(y)))
plt.xlabel('x')
plt.ylabel('rayon')

# Show the plot
plt.show()


print("liste des rayons" ,rayons(pressions(y)))

