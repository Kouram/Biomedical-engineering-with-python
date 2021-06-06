# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:04:56 2019

@author: kouram
"""

import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


#r est la masse volumique
r=1
#u est la visquosité dynamique
u=1
#notre précision
n=50
#valeurs du système
Pe=520 #pression d'entrée
Ps=10 # pression de sortie
l=20 #longueur du tube
Dmax=2 #diamètre maximal
C=1 #constante de la loi des tubes
po=1000 #terme en dénominateur dans l'exponentielle (loi des tubes)
phi = 5 #valeur initialisée du flux

A=8*r/mt.pi
B=128*u/mt.pi

#diamètre en fonction de la pression
def D(x):
    return Dmax*(1/(1+C*(mt.e**(-x/po))))

#dérivée de diamètre par rapport à la pression
def Ddev(x):
    return C*(Dmax/po)*(mt.e**(-x/po))/((1+C*(mt.e**(-x/po)))**2)


#fonction discrétisée dont on cherchera les racines par la méthode de Newton-Raphson
def F(p,P,ph):
    return P-p+A*(ph**2)*((1/D(P)**4)-(1/D(p)**4))+(B*ph*(D(p)**(-4))*(l/n))

#dérivée partielle de F par rapport à la première variable p (c-à-d : Pi)
def Fi_Piplus(p,P,ph):
    return 1 - 32*r*(ph**2)*Ddev(P)/(mt.pi*D(P)**5)

#dérivée partielle de F par rapport au flux
def Fi_ph(p,P,ph):
    return (16*r/mt.pi*D(P)**4)*ph - 16*r*ph/((mt.pi**2)*D(p)**4) + (128*u/(mt.pi*(D(p)**4)))*(l/n)

#dérivée partielle de F par rapport à la deuxième variable P (c-à-d : P(i+1))
def Fiplus_Piplus(p,P,ph):
    return -1 + 32*r*((ph**2)/mt.pi*(D(P)**5))*Ddev(P) - ((512*u*ph*Ddev(P))/(mt.pi*(D(P)**5)))*(l/n)



"""initialisation de la liste des pressions"""
p=[]
for i in range(1,n+1):
    a=(Ps-Pe)/(n-1)
    b=((n*Pe)-Ps)/(n-1)
    p.append(a*i+b)

"""initialisation du vecteur colonne des fonctions F noté Farray"""

Farray = np.zeros(shape=(n-1,1))
for i in range(n-1):
    Farray[i][0] = -F(p[i],p[i+1],phi)

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


"""implémentation de la boucle while qui raffine les valeurs des pressions et du flux"""
while np.linalg.norm(Farray) > 0.0001 :
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

print("x = " , x)#vecteur des erreurs finales
print("y = " , y)#vecteur des valeurs raffinées des pressions et flux


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

"""liste des diamètres"""
def Diamètres(f):
    h = []
    for i in range(len(f)):
        h.append(D(f[i]))
    return h

"""liste des rayons"""
def rayons(f):
    h=[]
    for i in range(len(f)):
        h.append(0.5*D(f[i]))
    return h


"""traçer P en fonction de z"""
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

"""traçer R le rayon en fonction de z"""
# Plot the data
plt.plot(x,rayons(pressions(y)))
plt.xlabel('x')
plt.ylabel('rayon')

# Show the plot
plt.show()

"""liste des rayons"""
print("liste des rayons" ,rayons(pressions(y)))

#vérification de l'équation scalaire pour deux valeurs successives de pression

def integrand(x,C,po):
     return (Dmax*(1/(1+C*(mt.e**(-x/po)))))**4

for i in range(len(pressions(y))-1):
    I = quad(integrand, pressions(y)[i], pressions(y)[i+1], args=(C,po))
    print(I[0]-32*(phi**2)*(1/mt.pi**2)*mt.log(D(pressions(y)[i+1])/D(pressions(y)[i])) + (128*u*phi*(l/n)/(mt.pi)))
