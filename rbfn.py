# Importar librerías
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def calculo_mse(cant_entradas,cant_neuronas,centroides,x,y,x_test,y_test):
    """
    Arguments: \n
    x = entrada de la red\n
    y = valores de esas entradas\n
    x_test = entrada para testing\n
    y_test = salida esperada\n
    """
    mse = 0;
    p = cant_entradas
    k = cant_neuronas
    # Calcular el sigma
    sigma = (max(centroides)-min(centroides))/np.sqrt(2*k)
    sigma = sigma[0]
    
    # Calcular matriz G
    G = np.zeros((p,k))
    for i in range(p):
        for j in range(k):
            dist = np.linalg.norm(x[i]-centroides[j], 2) # Distancia euclideana Entre Xi y Cj
            G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij

    W = np.dot(np.linalg.pinv(G), y.T)

    p = len(x_test)
    # Propagar la red
    G = np.zeros((p,k))
    for i in range(p):
        for j in range(k):
            dist = np.linalg.norm(x_test[i]-centroides[j], 2) # Distancia euclideana Entre Xi y Cj
            G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij
    ynew = np.dot(G, W) # Salida de la red

    #mse = np.square(np.subtract(y_test,ynew)).mean()
    mse = np.square(np.subtract(y_test[-(len(y_test)-len(y)):],ynew[-(len(y_test)-len(y)):])).mean()

    return mse







def plotear_predicha(cant_entradas,cant_neuronas,centroides,x,y,x_test,y_test):
    
    p = cant_entradas
    k = cant_neuronas
    # Calcular el sigma
    sigma = (max(centroides)-min(centroides))/np.sqrt(2*k)
    sigma = sigma[0]
    # Calcular matriz G
    G = np.zeros((p,k))
    for i in range(p):
        for j in range(k):
            dist = np.linalg.norm(x[i]-centroides[j], 2) # Distancia euclideana Entre Xi y Cj
            G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij

    W = np.dot(np.linalg.pinv(G), y.T)

    p = len(x_test)
    # Propagar la red
    G = np.zeros((p,k))
    for i in range(p):
        for j in range(k):
            dist = np.linalg.norm(x_test[i]-centroides[j], 2) # Distancia euclideana Entre Xi y Cj
            G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij
    ynew = np.dot(G, W) # Salida de la red
    # Dibujar puntos
    plt.plot(x_test,ynew, '-b')
    plt.plot(x_test,y_test, '-r')
    plt.show()





"""
# Crear problema (Espacio de entrada y salida esperada)
p = 15 # Número de muestras
x = np.linspace(-5, 5, p).reshape(1,-1)
y = 2 * np.cos(x) + np.sin(3*x) + 5

# Dibujar puntos
plt.plot(x,y, 'or')

# Definir número de neuronas
k = 14

# Agrupar puntos en clústers
model = KMeans(n_clusters=k)
model.fit(x.T)

# Extraer centroides
c = model.cluster_centers_

# Calcular el sigma
sigma = (max(c)-min(c))/np.sqrt(2*k)
sigma = sigma[0]

# Calcular matriz G
G = np.zeros((p,k))
for i in range(p):
    for j in range(k):
        dist = np.linalg.norm(x[0,i]-c[j], 2) # Distancia euclideana Entre Xi y Cj
        G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij

W = np.dot(np.linalg.pinv(G), y.T)

# Propagar la red
p = 200
xnew = x = np.linspace(-5, 5, p).reshape(1,-1)

G = np.zeros((p,k))
for i in range(p):
    for j in range(k):
        dist = np.linalg.norm(x[0,i]-c[j], 2) # Distancia euclideana Entre Xi y Cj
        G[i,j] = np.exp((-1/(sigma**2))*dist**2) # Resultado de la función de activación para Gij

ynew = np.dot(G, W) # Salida de la red
# Dibujar puntos
plt.plot(xnew.T, ynew, '-b')
plt.show()
"""