import numpy as np
import scipy.optimize as op

import scipy.integrate as integrate

import numpy as np

import math

import matplotlib.pyplot as plt


h1=0.88

h2=0.68

d1=0.02

l1=0.135

u=1.85*(10**(-5))

pc=590

p0=1.013*(10**5)

p_alv_max=15000

p_alv=p_alv_max+p0

rho=1.014

dt=4/129

RV=0.0015

CV=0.00561

VL=0.0075


Const=(VL-RV)/CV


def P_colonne(n):

nbr_de_noeuds=2**(n+1)

P=nbr_de_noeuds*(lambda p: p)# P est la colonne des pressions à chaque noeud, incluant p0 et palv

for k in range(2**n,len(P)):# On remplace dans P les valuers connues

P[k]=p_alv

P[0]=p0

return P


def nombre_de_branches(n):

s=0

for i in range(n+1):

s+=2**i

return s


def h1_h2(n):

    return [0,1]+((nombre_de_branches(n)-1)//2)*[h1,h2]


def liste_diametres_max(n):

    Dmax=np.zeros(nombre_de_branches(n)+1)

    H=h1_h2(n)

    Dmax[1]=d1

    for i in range (2,nombre_de_branches(n)+1):

    Dmax[i]=H[i]*Dmax[i//2]

    return Dmax

def alpha(n):

    alph=[0,0.5,1/9,1/9] #+D[1]->0, le zero initial garanti l'égalité de longueur entre la liste Dmax et alph

    for i in range(4,nombre_de_branches(n)+1):

    m=int(math.log(i)/math.log(2))

    alph.append((m/3)**2)

    return alph


def colonne_Diametre(n):

colonne_D=np.zeros(nombre_de_branches(n)+1) #premier element nul pour garder les indices des equations

liste_Dmax=liste_diametres_max(n)

liste_alpha=alpha(n)

for j in range(nombre_de_branches(n)+1):

colonne_D[j]=lambda p: (liste_Dmax[j]*(1/(1+liste_alpha[j]*math.exp(-1*(p/pc)))))

return colonne_D

def matriceA_i_j(n):

P= P_colonne(n)

A=np.zeros((nbr_de_noeuds,nbr_de_noeuds))

for i in range (len(P)):

for j in range(1,len(P)):

if i==(j//2):

A[i][j]=integrate.quad(lambda p: (colonne_Diametre(p,n)[j])**4,P[j],P[i])[0]

return A

def matriceB_i_j(n):

P= P_colonne(n)

nbr_de_noeuds=2**(n+1)

B=np.zeros((nbr_de_noeuds,nbr_de_noeuds))

for i in range(len(P)):

for j in range(1,len(P)):

if i==(j//2):

B[i][j]=(32*rho/((np.pi)**2))*(math.log((colonne_Diametre(P[i],n)[j])/(colonne_Diametre(P[j],n)[j])))

return B

def liste_longueurs(n):

L=np.zeros(nombre_de_branches(n)+1)

H=h1_h2(n)

L[1]=l1

for i in range (2,nombre_de_branches(n)+1):

L[i]=H[i]*L[i//2]

return L


def colonne_C(n):

L=liste_longueurs(n)

C=np.zeros(nombre_de_branches(n)+1)

for i in range(len(L)):

C[i]=(128/np.pi)*u*L[i]

return C

def matricePHI_i_j(n):

A=matriceA_i_j(n)

B=matriceB_i_j(n)

C=colonne_C(n)

nbr_de_noeuds=2**(n+1)

PHI=np.zeros((nbr_de_noeuds,nbr_de_noeuds))

for i in range(len(P)):

for j in range(1,len(P)):

if i==(j//2):

PHI[i][j] = (C[j]-math.sqrt(C[j]**2+4*A[i][j]*B[i][j]))/(2*B[i][j])

return PHI

def pression_alv_t(m):

t=np.linspace(0,4,m)

p_alv_t=p_alv_max*(np.exp(-1*t))+m*[p0]

return (p_alv_t,t)

def f():

PHI=matricePHI_i_j(n)

nbr_de_noeuds=2**(n+1)

m=2**(n)-1#dernier père, nombre d'équation

list_equation_P=[]

for j in range(1,m+1):

list_equation_P.append(PHI[j][2*j]+PHI[j][2*j+1]-PHI[j//2][j])

list_equation_P=[0]+list_equation_P+(m+1)*[0]

return list_equation_P


n=input("entrer n: ")

n=int(n)

nbr_de_noeuds=2**(n+1)


tuple_palv_t = pression_alv_t(130)

p_alv_t=tuple_palv_t[0]

t=tuple_palv_t[1]

liste_flux_sortant=[]

volume_t=[0]


for i in range(130):

p_alv=p_alv_t[i]#la matrice A i j varie avec le temps selon palv, de meme pour B i j et C i j

P=op.fsolve(f,P0,maxfev=1000)

PHI_t=matricePHI_i_j(P,n)

flux_sortant_t=PHI_t[0][1]

liste_flux_sortant.append(flux_sortant_t)

volume_t.append(liste_flux_sortant[i]*dt)

volume_t[i+1]=volume_t[i]+volume_t[i+1]

liste_flux_sortant=[0]+liste_flux_sortant


plt.plot(volume_t,liste_flux_sortant)

plt.show()


plt.plot(t,liste_flux_sortant[1:])

plt.show()


