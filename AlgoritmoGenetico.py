import random
import rbfn

def seleccion(poblacion):
    #TODO hay que ordenar por fitness, agregamos gen?
    seleccionados = []
    seleccionados.append(poblacion[0])
    for n in range(1,len(poblacion)):
        seleccionados.append(poblacion[random.randrange(n)])
    return seleccionados

def cruza_uniforme(individuo_1, individuo_2):
    hijo_1 = {
        "entradas": 0,
        "hidden": 0,
        "centers": []
    }
    hijo_2 = {
        "entradas": 0,
        "hidden": 0,
        "centers": []
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