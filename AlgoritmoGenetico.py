import random
import rbfn
import numpy as np
from sklearn.cluster import KMeans

def seleccion(poblacion):
    seleccionados = []
    seleccionados.append(poblacion[0])
    for n in range(1,len(poblacion)):
        seleccionados.append(poblacion[random.randrange(n)])
        # Para evitar cruzar un individuo consigo mismo
        while (n%2 == 0 and poblacion[n-1]["entradas"] == poblacion[n]["entradas"]
            and poblacion[n-1]["hidden"] == poblacion[n]["hidden"]):
            seleccionados[n] = poblacion[random.randrange(n)]
    return seleccionados

def cruza_uniforme(individuo_1, individuo_2):
    hijo_1 = {
        "entradas": 0,
        "hidden": 0,
        "centers": [],
        "mse" : -1.0
    }
    hijo_2 = {
        "entradas": 0,
        "hidden": 0,
        "centers": [],
        "mse" : -1.0
    }
    mask = random.randrange(25) # Numero entero entre 0 y 25
    # Calculo del primer hijo
    # Aplico mascara para un individuo y el complemento de la mascara al otro y sumo
    entradas_aux = (individuo_1["entradas"] & mask) + (individuo_2["entradas"] & ~mask)
    hidden_aux = (individuo_1["hidden"] & mask) + (individuo_2["hidden"] & ~mask)
    # Linea para la RBF consistente -> neuronas <= inputs
    if (hidden_aux > entradas_aux): hidden_aux =hidden_aux & entradas_aux
    hijo_1["entradas"] = entradas_aux
    hijo_1["hidden"] = hidden_aux 
    
    # Calculo del segundo hijo
    entradas_aux = (individuo_1["entradas"] & ~mask) + (individuo_2["entradas"] & mask)
    hidden_aux = (individuo_1["hidden"] & ~mask) + (individuo_2["hidden"] & mask)
    if (hidden_aux > entradas_aux): hidden_aux =hidden_aux & entradas_aux
    hijo_2["entradas"] = entradas_aux
    hijo_2["hidden"] = hidden_aux 
    
    return hijo_1,hijo_2

def mutacion(individuo):
    mask = random.randrange(25) # Numero entero entre 0 y 25
    # Aplico mascara para un individuo y el complemento de la mascara al otro y sumo
    entradas_aux = (individuo["entradas"] & ~mask) + (~individuo["entradas"] & mask)
    hidden_aux = (individuo["hidden"] & ~mask) + (~individuo["hidden"] & mask)
    # Linea para la RBF consistente -> neuronas <= inputs
    if (hidden_aux > entradas_aux): hidden_aux = hidden_aux & entradas_aux
    individuo["entradas"] = entradas_aux
    individuo["hidden"] = hidden_aux
    return individuo

# Funcion que utiliza para ordenar la lista
def fitness_sort(e):
    return e["mse"]

### CODIGO ###

# Definir la funcion que vamos a usar para prueba
x = np.linspace(-5, 5, 50).reshape(1,-1)
y = 2 * np.cos(x) + np.sin(3*x) + 5

# Variables
poblacion = []
tam_poblacion = 50
maximo_entradas = 20 # Esto va a depender de cuantos puntos definamos en la funcion

# Definir poblacion inicial
for i in range(1,tam_poblacion):
    aux = {
        "entradas": 1,
        "hidden": 1,
        "centers": [],
        "mse" : -1.0
    }
    aux_entradas = random.randrange(maximo_entradas)
    aux_hidden = random.randrange(aux_entradas)
    aux["entradas"] = aux_entradas # Asigno entradas
    aux["hidden"] = aux_hidden # Asigno neuronas
    # Calculo de los clusters del individuo
    model = KMeans(n_clusters= aux["hidden"])
    model.fit(x.T[-aux["entradas"]:]) #TODO Ver esta sintaxis, esta es la parte complicada
    aux["centers"] = model.cluster_centers_ # Asigno centroides
    poblacion.append(aux)
    
# Calcular el fitness de la poblacion
for individuo in poblacion:
    individuo["mse"] = rbfn.calculo_mse(individuo["entradas"],individuo["hidden"],individuo["centers"],
        x[-individuo["entradas"]:],y[-individuo["entradas"]:]) #TODO ver como pasar el x e y, parte complicada x2
    
# Ordenar poblacion por fitness
poblacion.sort(key=fitness_sort)

# Inicio Bucle - Condicion de finalizacion: la solucion no mejora en 3 iteraciones
mejor_individuo = poblacion[0] # Almacenar el mejor individuo de la poblacion (solo inicializo)
mejor_solucion = 0 # Cuenta las iteraciones sin una mejor solucion

while mejor_solucion < 3 :        
    
    # Realizar seleccion
    seleccion = seleccion(poblacion)
    
    # Cruzar seleccion 
    hijos = []
    for i in range(1,len(seleccion)):
        hijos.append(cruza_uniforme(seleccion[i-1],seleccion[i]))
        i+=1
    
    # Probabilidad de mutacion
    if (random.random() > 0.8) :
        mutacion(hijos[random.randrange(len(hijos)-1)]) #TODO revisar este indice
    
    # Obtener centroides (model.get sentros)
    for individuo in hijos:
        model = KMeans(n_clusters= individuo["hidden"])
        model.fit(x.T[-individuo["entradas"]:]) #TODO Ver esta sintaxis, esta es la parte complicada
        individuo["centers"] = model.cluster_centers_ # Asigno centroides
    
    # Calcular el fitness de los hijos
    for individuo in poblacion:
        individuo["mse"] = rbfn.calculo_mse(individuo["entradas"],individuo["hidden"],individuo["centers"],
            x[-individuo["entradas"]:],y[-individuo["entradas"]:]) #TODO ver como pasar el x e y, parte complicada x2
    
    # Crear nueva poblacion
    poblacion.extend(hijos) # Agrego los hijos
    poblacion.sort(key=fitness_sort) # Ordeno por fitness
    poblacion = poblacion[:tam_poblacion] # Trunco la poblacion
    
    # Evaluo el mejor individuo para la condicion de parada
    if ((mejor_individuo["mse"] > poblacion["mse"]) or (mejor_individuo["mse"] == -1.0)) :
        mejor_individuo = poblacion[0]
        mejor_solucion = 0
    else:
        mejor_solucion+=1
    # Repite
print(mejor_individuo)