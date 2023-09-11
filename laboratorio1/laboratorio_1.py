#Javier Alejandro Mazariegos Godoy 
#20200223
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import cvlib
from random import randrange
import sys

def imgpad(img, r):
    """Agrega un borde de píxeles en negro alrededor de una imagen.

    Args:
        img (numpy.ndarray): Una matriz numpy que representa la imagen de entrada.
        r (int): El número de píxeles en negro que se agregarán al borde de la imagen.

    Returns:
        numpy.ndarray: Una nueva matriz numpy que representa la imagen con el borde agregado.
    """
    columns, rows = img.shape
    rows_insert = [255] * rows   #Anotacion: Se cambió a 255 debido a que el borde oscuro no se mira por la imagen que tiene el mismo fondo.
    for x in range(r):
        img = np.insert(img,0, rows_insert,axis=0)
        img = np.insert(img,img.shape[0], rows_insert,axis=0)
    column_insert = [255] * img.shape[0] #Anotacion: Se cambió a 255 debido a que el borde oscuro no se mira por la imagen que tiene el mismo fondo.
    for x in range(r):
        img = np.insert(img,0, column_insert,axis=1)
        img = np.insert(img,img.shape[1],column_insert,axis=1)
    return img

#Se encuentra el padre de cada label
def find(data, i):
    """Encuentra el padre de un elemento en una estructura de datos de conjunto disjunto (Union-Find).

    Args:
        data (List[int]): Una lista que representa una estructura de datos de conjunto disjunto (Union-Find).
        i (int): El elemento cuyo padre se debe encontrar.

    Returns:
        int: El radre del elemento 'i' en la estructura de conjunto disjunto.
    """
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]

#Se genera las relaciones entre los labeles.
def union(data, i, j):
    """Une dos conjuntos en una estructura de datos de conjunto disjunto (Union-Find).

    Args:
        data (List[int]): Una lista que representa una estructura de datos de conjunto disjunto (Union-Find).
        i (int): El primer elemento del conjunto a unir.
        j (int): El segundo elemento del conjunto a unir.
    """
    pi, pj = find(data, i), find(data, j)
    #Se relaizó un cambio para que se encontrara el padre más pequeño.
    if pi != pj:
        if(pi < pj):
            data[pj] = pi
        else:
            data[pi] = pj

def connected_c(img):
    """Esta funcion se encarga de etiquetar los componentes conectados de una imagen binaria. 
    Primero se realiza el proceso de First pass asignando las etiquetas y luego el Second pass encontrando las etiquetas padres.

    Args:
        img (numpy.ndarray): Una matriz numpy que representa la imagen binaria de entrada.

    Returns:
        numpy.ndarray: Una nueva matriz numpy con las etiquetas de componentes conectados asignadas a cada píxel.
    """
    #First_pass
    tuplas_conecciones = []
    img2 = np.zeros(img.shape, dtype=int)
    keys_count = 0
    #Se recorre fila,columna
    for fila in (range(img.shape[0])):
        for columna in (range(img.shape[1])):
            if(img[fila,columna] != 0):  #Paso1: consiste en comprobar, si estamos interesados en un píxel o no.

                #Segundo paso: Obtenemos las etiquetas de los píxeles de arriba y a la izquierda de p.
                #Aquí se crean las nuevas etiquetas
                if((fila == 0 and columna == 0) or (img[fila, columna-1] == 0 and fila == 0) or (img[fila-1,columna] ==0 and columna ==0) or (img[fila-1, columna] == 0 and img[fila, columna-1] == 0)):
                    keys_count = keys_count + 1
                    img2[fila,columna] = keys_count
                #Aquí se asignan las etiquetas que ya están creadas
                else:
                    #Si estoy en la primera fila o la fila de arriba tiene un 0, entonces asigno la etiqueta que tiene el vecino izquierdo
                    if(fila == 0 or img[fila-1, columna] == 0):
                        img2[fila,columna] = img2[fila, columna-1]
                    #Si estoy en la primera columna o la columna de la iquierda es un 0, entonces asigno la etiqueta que tiene el vecino de arriba.
                    elif(columna == 0 or img[fila, columna-1] == 0):
                        img2[fila,columna] = img2[fila-1, columna]
                    #Situacion en la que ambos vecinos son distintos de 0.
                    else:
                        #situación 1: Los dos vecinos tienen la misma etiqueta.
                        if(img2[fila-1, columna] == img2[fila, columna-1]):
                            img2[fila,columna] = img2[fila-1, columna]
                        #situacion2: Los dos veciones tienen etiquetas distintas, enyonces se asigna la etiqueta más pequeña. 
                        else:
                            img2[fila,columna] = min(img2[fila-1, columna], img2[fila, columna-1])
                            tuplas_conecciones.append((img2[fila, columna-1],img2[fila-1, columna]))
    #Second_pass
    img3 = img2.copy()
    #Primero se llama a union para que haga todas las relaciones con las tuplas que se generarón en first pass
    datos_conecciones = [i for i in range(keys_count+1)]
    for i, j in tuplas_conecciones:
        union(datos_conecciones, i, j)
    #Luego se recorre la matriz y se llama a find, para encontrar el padre de cada label. 
    for fila in (range(img3.shape[0])):
        for columna in (range(img3.shape[1])):
            if(img3[fila,columna] != 0):
                img3[fila, columna] = find(datos_conecciones, img3[fila, columna])
    return img3

def labelview(labels):
    """Esta funcion permite asignar un color a cada etiqueta de la matriz de labels y visualizarla.

    Args:
         labels (numpy.ndarray): Una matriz numpy que contiene las etiquetas de componentes conectados.
    """
    global outpu_img
    #Se crea una matriz de ceros y luego a color para poderselo a signar a cada pixel. 
    img3 =  np.zeros(labels.shape, dtype=np.uint8)
    img3 = cv.cvtColor(img3, cv.COLOR_GRAY2RGB)
    colores = {}
    #Se recorre la matriz y se asigna un color a cada label
    for fila in (range(labels.shape[0])):
        for columna in (range(labels.shape[1])):
            if(labels[fila,columna] != 0):
                if(labels[fila,columna] in colores.keys()):
                    img3[fila,columna] = colores[labels[fila,columna]]
                else:
                    rand_color = (randrange(255), randrange(255), randrange(255))
                    colores[labels[fila,columna]] = rand_color
                    img3[fila,columna] = rand_color
    cvlib.imgview(img3, filename=outpu_img)


if(len(sys.argv) >= 2):
    input_img = sys.argv[1]
    outpu_img = sys.argv[2]
    img = cv.imread(input_img, cv.IMREAD_GRAYSCALE)
    if(len(sys.argv) >= 4 ): #Si se desea ejecutar la función imgpad usar el comando: python laboratorio1.py fprint3.pgm fprint3_ccl.png imgpad 10
        cvlib.imgview(imgpad(img, int(sys.argv[4])))
    else:
        img_blur = cv.GaussianBlur(img,(3,3),0)
        binarizacion = 255- cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,21,15)
        labelview(connected_c(binarizacion))
else:
    print("Debe de ingresar todos los parametros: laboratorio_1.py fprint3.pgm fprint3_ccl.png")
