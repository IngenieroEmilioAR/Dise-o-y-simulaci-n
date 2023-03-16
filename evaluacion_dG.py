import numpy as np
from sympy import symbols, integrate, plot

import matplotlib.pyplot as plt


T, C1, C2 = symbols('T, C1, C2')

R = 1.9872


#Funciones para el cálculo de la entalpía, constantes de integración y energía ibre de gibbs.

def cp(a0, a1, a2, a3, a4):
    """
    Creación de una expresión simbólica para el Cp.
    """
    return a0 + a1*T + a2*T**2 + a3*T**3 +a4*T**4

def H(cp):
    """
    Función que recibe una expresión de Cp y la integra indefinidamente 
    respecto a T
    """
    return integrate(cp, T) + C1

def eval_c1(H, dHf, Tref):
    """
    Evaluación de la constante de integración, empleando la entalpía de 
    formación a la temperatura de referencia (298.15K)
    """

    return -(H.subs(T, Tref) - C1 - dHf)

def dG_T(H):
    return -integrate(H/(T**2), T) +C2

def eval2_c2(G_T, dGf, Tref):
    return -(G_T.subs(T, Tref) - dGf/Tref - C2)

def dG(G_T, Ci2):
    return G_T.subs(C2,Ci2)*T





def secuencia(coef_cp, dHref: list, dGref: list, Tref = 298.15):
    
    
    cp1 = cp(coef_cp[0][0], coef_cp[0][1], coef_cp[0][2], coef_cp[0][3], coef_cp[0][4])
    cp2 = cp(coef_cp[1][0], coef_cp[1][1], coef_cp[1][2], coef_cp[1][3], coef_cp[1][4])
    cp3 = cp(coef_cp[2][0], coef_cp[2][1], coef_cp[2][2], coef_cp[2][3], coef_cp[2][4])
    
    H1 = H(cp1)
    H2 = H(cp2)
    H3 = H(cp3)
    
    c1_1 = eval_c1(H1, dHref[0], Tref)
    c1_2 = eval_c1(H2, dHref[1], Tref)
    c1_3 = eval_c1(H3, dHref[2], Tref)

    #print(c1_1, c1_2, c1_3)
    
    H1 = H1.subs(C1,c1_1)
    H2 = H2.subs(C1,c1_2)
    H3 = H3.subs(C1,c1_3)
    
    G_T1 = dG_T(H1)
    G_T2 = dG_T(H2)
    G_T3 = dG_T(H3)
    
    c2_1 = eval2_c2(G_T1, dGref[0], Tref)
    c2_2 = eval2_c2(G_T2, dGref[1], Tref)
    c2_3 = eval2_c2(G_T3, dGref[2], Tref)
    
    G1 = dG(G_T1, c2_1)
    G2 = dG(G_T2, c2_2)
    G3 = dG(G_T3, c2_3)
    
    return G1, G2, G3, G3-G2

def evaluar(rango, funciones):
    A = []
    for i in funciones:
        B = []
        for j in rango:
            a = i.subs(T,j).simplify()
            B.append(a)
        A.append(B)
    return np.array(A)










#Reacción F2 + H2 -> HF
dHref1 = [0,0,-273.3*1000/4.184]
dGref1 = [0,0, -275.4*1000/4.18]

G1, G2, G3, G3_G2 = secuencia(
                         [np.array([3.347, 0.467e-3, 0.526e-5, 0.794e-8, 0.33e-11])*R,
                         np.array([2.883, 3.681e-3, -0.772e-5, 0.692e-8, -0.213e-11])*R,
                         np.array([3.901, 3.708e-3, 1.165e-5, 1.465e-8, 0.639e-11])*R],
                         dHref1, dGref1)

#print(G1, '\n\n', G2, '\n\n', G3)

temp = np.linspace(1500,2500)

Y = evaluar(temp, [G1,G3_G2])


fig, ax = plt.subplots()

ax.plot(temp, Y[0], label = "F2")
ax.plot(temp, Y[1], label = "HF - H2")





#Reacción F2 + 2NO2 -> 2NO2F

dHref1 = [0, 9.16*1000/4.184,-108.78*1000/4.184]
dGref1 = [0, 97.85*1000/4.184, -66.55*1000/4.18]



G1, G2, G3, G3_G2 = secuencia(
                        [np.array([3.347, 0.467e-3, 0.526e-5, 0.794e-8, 0.33e-11])*R, 
                         np.array([3.374, 27.257e-3, -1.917e-5, -0.616e-8, 0.859e-11])*R, 
                         np.array([1.620, 20.883e-3, -2.512e-5, 1.586e-8, -0.42e-11])*R],
                         dHref1, dGref1)

Y = evaluar(temp, [G3_G2])
ax.plot(temp, Y[0], label = "2NO2F - 2NO2")







#Reacción F2 + Ca -> CaF2 (del Perry)

dHref1 = [0, 0*1000/4.184, -286.5e3]
dGref1 = [0, 0*1000/4.184, -264.1e3]

G1, G2, G3, G3_G2 = secuencia(
                        [np.array([3.347, 0.467e-3, 0.526e-5, 0.794e-8, 0.33e-11])*R,
                        [5.31, 0.00333, 0, 0, 0], 
                        [14.7, 0.0038, 0, 0, 0]],
                         dHref1, dGref1)

Y = evaluar(temp, [G1,G3_G2])
ax.plot(temp, Y[1], label = "CaF2 - Ca")






#Reacción F2 + 2K -> 2KF (del Perry)

dHref1 = [0, 0*1000/4.184, -138.36e3]
dGref1 = [0, 0*1000/4.184, -133.13e3]

G1, G2, G3, G3_G2 = secuencia(
                        [np.array([3.347, 0.467e-3, 0.526e-5, 0.794e-8, 0.33e-11])*R, 
                         [5.24, 0.00555, 0, 0, 0], 
                         [10.8, 0.00284, 0, 0, 0]],
                         dHref1, dGref1)

Y = evaluar(temp, [G1,G3_G2])
#plt.plot(temp, Y[0])
ax.plot(temp, Y[1], label = "2KF - 2K")







#Reacción F2 + 2Na -> 2NaF (del Perry)

dHref1 = [0, 0*1000/4.184, -135.94e3]
dGref1 = [0, 0*1000/4.184, -129e3]

G1, G2, G3, G3_G2 = secuencia(
                        [np.array([3.347, 0.467e-3, 0.526e-5, 0.794e-8, 0.33e-11])*R, 
                         [5.01, 0.00536, 0, 0, 0], 
                         [10.4, 0.00289, 0, 0, 0]],
                         dHref1, dGref1)

Y = evaluar(temp, [G1,G3_G2])
#plt.plot(temp, Y[0])
ax.plot(temp, Y[1], label = "2NaF - 2Na")







ax.legend()

ax.set_xlabel("Temperatura [K]")
ax.set_ylabel(" $\Delta$G [cal/mol]")

ax.grid()
plt.show()


