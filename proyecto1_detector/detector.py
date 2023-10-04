#Javier Alejandro Mazariegos Godoy 20200223
import cvlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
from joblib import load

def number2char(num):
    """
    Convierte un número en un carácter según la siguiente correspondencia:
    
    - 0 a 9 se mapean a los números 0 a 9 respectivamente.
    - 10 a 34 se mapean a las letras A a Z respectivamente.
    
    Args:
        num (int): El número entero que se desea convertir en un carácter.
        
    Returns:
        str or int: El carácter correspondiente al número de entrada. Si el número está fuera del rango
                    de 0 a 34, se devuelve el mismo número de entrada.
    """
    if(num == 0):
        return 0
    elif(num == 1):
        return 1
    elif(num == 2):
        return 2
    elif(num == 3):
        return 3
    elif(num == 4):
        return 4
    elif(num == 5):
        return 5
    elif(num == 6):
        return 6
    elif(num == 7):
        return 7
    elif(num == 8):
        return 8
    elif(num == 9):
        return 9
    elif(num == 10):
        return 'A'
    elif(num == 11):
        return 'B'
    elif(num == 12):
        return 'C'
    elif(num == 13):
        return 'D'
    elif(num == 14):
        return 'E'
    elif(num == 15):
        return 'F'
    elif(num == 16):
        return 'G'
    elif(num == 17):
        return 'H'
    elif(num == 18):
        return 'I'
    elif(num == 19):
        return 'J'
    elif(num == 20):
        return 'K'
    elif(num == 21):
        return 'L'
    elif(num == 22):
        return 'M'
    elif(num == 23):
        return 'N'
    elif(num == 24):
        return 'P'
    elif(num == 25):
        return 'Q'
    elif(num == 26):
        return 'R'
    elif(num == 27):
        return 'S'
    elif(num == 28):
        return 'T'
    elif(num == 29):
        return 'U'
    elif(num == 30):
        return 'V'
    elif(num == 31):
        return 'W'
    elif(num == 32):
        return 'X'
    elif(num == 33):
        return 'Y'
    elif(num == 34):
        return 'Z'

def leer_imagenes(indice_imagen):
    """
    Lee una imagen desde el path especificado y realiza transformaciones en la imagen.

    Esta función carga una imagen desde el path especificado utilizando OpenCV (cv2). 
    Luego, se convierte la imagen de formato BGR a RGB y crea una versión en escala de grises.

    Args:
        indice_imagen (str): El nombre del archivo de la imagen que se desea leer.

    Returns:
        tuple: Una tupla que contiene dos elementos:
            - La imagen en formato RGB.
            - La imagen en escala de grises.
    """
    img = cv.imread(indice_imagen,cv.IMREAD_COLOR) 
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray

def binarizacion(img, tresh):
    """
    Aplica un proceso de binarización adaptativa a una imagen.

    Esta función utiliza la binarización adaptativa con umbral gaussiano inverso en una imagen. 
    Además, verifica si el histograma de la imagen original tiene más píxeles oscuros que claros 
    y, si es así, invierte los valores binarizados.

    Args:
        img (numpy.ndarray): La imagen de entrada en formato numpy array.
        tresh (int): El valor del umbral para la binarización adaptativa.

    Returns:
        numpy.ndarray: La imagen binarizada después del procesamiento.
    """
    binarized2 = cv.adaptiveThreshold(img, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,tresh,5)
    histr = cv.calcHist([binarized2],[0],None,[256],[0,256])
    if (histr[0][0] < histr[255][0]):
        binarized2 = 255 -binarized2
    return binarized2

def encontrar_contornos(img):
    """
    Encuentra los contornos en una imagen binarizada.

    Esta función utiliza la función `cv.findContours` de OpenCV para encontrar y obtener los contornos 
    de una imagen binarizada. Los contornos se encuentran utilizando el modo `cv.RETR_TREE` y el método 
    `cv.CHAIN_APPROX_SIMPLE`.

    Args:
        img (numpy.ndarray): La imagen binarizada en formato numpy array.

    Returns:
        list numpy.ndarray: Una lista de contornos encontrados en la imagen. 
                                Cada contorno es una matriz numpy que representa un conjunto de puntos.
    """
    mode = cv.RETR_TREE
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] 
    return cv.findContours(img, mode, method[1])

def encontrar_placa(binarized2, contours, hierarchy):
    """
    Encuentra y valida posibles placas dentro de una imagen binarizada.

    Esta función analiza la imagen binarizada junto con sus contornos y jerarquía de contornos para encontrar
    posibles placas en la imagen. Se aplican criterios de selección, como la existencia de al menos un hijo, 
    el aspect ratio y la extensión del contorno. Si se encuentra una posible placa, se devuelve la información
    relacionada con la placa y su imagen.

    Args:
        binarized2 (numpy.ndarray): La imagen binarizada en formato numpy array.
        contours (list numpy.ndarray): Lista de contornos encontrados en la imagen.
        hierarchy (numpy.ndarray): Jerarquía de contornos.

    Returns:
        tuple: Una tupla que contiene información sobre la posible placa o un indicador de que no se encontró una.
            - bandera_siguiente (int): Un indicador (0 o 1) que indica si se encontró una posible placa.
            - x1 (int): Coordenada x del rectángulo de la posible placa.
            - y1 (int): Coordenada y del rectángulo de la posible placa.
            - w1 (int): Ancho del rectángulo de la posible placa.
            - h1 (int): Altura del rectángulo de la posible placa.
            - placa (numpy.ndarray): La imagen de la posible placa si se encuentra, de lo contrario, una cadena vacía.
    """
    contornos_posibles = {}          
    bandera_siguiente = -1
    r = binarized2.copy()
    for c in range(len(contours)):
        contour = contours[c]
        
        # Verificar si el contorno tiene al menos un hijo en la jerarquía.
        num_children = 0
        for hierarchy_entry in hierarchy[0]:
            if hierarchy_entry[3] == c:
                num_children += 1

        if(num_children >= 1):
            # Crear una máscara para aislar el área del contorno.
            mask = np.zeros_like(binarized2)
            cv.drawContours(mask, contours, c, (255, 255, 255), thickness=cv.FILLED)
            region_of_interest = cv.bitwise_and(r, mask)
            
             # Obtener el rectángulo que encierra el contorno.
            x, y, w, h = cv.boundingRect(contours[c])
            img_new = region_of_interest[y:y+h, x:x+w]

            #Validar la proporcion, mayor width que height
            aspect_ratio = float(w)/h
            if(aspect_ratio > 1.5):
                area = cv.contourArea(contours[c])
                x,y,w,h = cv.boundingRect(contours[c])
                rect_area = w*h
                #Validar que la extension sea mayor a 0.8, es decir que sea un cuadrado
                extent = float(area)/rect_area
                if(extent > 0.8):
                    #Se obtiene el contorno con mayor extent
                    if(len(contornos_posibles.keys()) > 0):
                        if(list(contornos_posibles.keys())[0] < extent):
                            contornos_posibles.clear()
                            contornos_posibles[extent] = [img_new, (x,y,w,h)]
                    else:
                        contornos_posibles[extent] = [img_new, (x,y,w,h)]

    # Validar si contornos_posibles está vacío.
    if(len(contornos_posibles.keys()) == 0):
        bandera_siguiente = 0
    else:
        bandera_siguiente = 1
        placa = list(contornos_posibles.values())[0][0]

    if(bandera_siguiente == 1):
        x1,y1,w1,h1  = list(contornos_posibles.values())[0][1]
        return (bandera_siguiente,x1,y1,w1,h1,placa)
    else:
        return (bandera_siguiente,"","","","","")

def encontrar_caracteres(img, binarized2, contours, hierarchy, bandera_siguiente, x1,y1, model):
    """
    Encuentra y reconoce caracteres dentro de una imagen binarizada y devuelve los resultados.

    Esta función analiza una imagen binarizada junto con sus contornos, jerarquía de contornos y otros parámetros.
    Se buscan y reconocen caracteres dentro de los contornos seleccionados y se devuelve la imagen original con
    los caracteres reconocidos y los resultados como una cadena.

    Args:
        img (numpy.ndarray): La imagen original en formato numpy array.
        binarized2 (numpy.ndarray): La imagen binarizada en formato numpy array.
        contours (list of numpy.ndarray): Lista de contornos encontrados en la imagen.
        hierarchy (numpy.ndarray): Jerarquía de contornos.
        bandera_siguiente (int): Un indicador (0 o 1) que indica si se encontró una posible placa.
        x1 (int): Coordenada x del rectángulo de la posible placa.
        y1 (int): Coordenada y del rectángulo de la posible placa.
        model (objeto): El modelo utilizado para reconocer caracteres en las imágenes.

    Returns:
        tuple: Una tupla que contiene la imagen original con los caracteres reconocidos y los resultados como una cadena.
            - imagen_final (numpy.ndarray): La imagen original con los caracteres reconocidos dibujados.
            - resultados (str): Los caracteres reconocidos en la imagen.
    """
    contornos_dict = {}
    imagen_final = img.copy()
    r = binarized2.copy()
    text = []

    for c in range(len(contours)):
        contour = contours[c]

        # Verificar si el contorno tiene como máximo 2 hijos en la jerarquía.
        num_children = 0
        for hierarchy_entry in hierarchy[0]:
            if hierarchy_entry[3] == c:
                num_children += 1

        if(num_children <= 2):
            # Crear una máscara para aislar el área del contorno.
            mask = np.zeros_like(binarized2)
            cv.drawContours(mask, contours, c, (255, 255, 255), thickness=cv.FILLED)
            region_of_interest = cv.bitwise_and(r, mask)

            # Se crea una mascara para el rectangulo que encierra el contorno
            x,y,w,h = cv.boundingRect(contours[c])
            rect_image = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contours[c] - (x, y)
            cv.drawContours(rect_image, [shifted_contour], 0, 255, thickness=cv.FILLED)
            rect_image = 255 - rect_image
            region_of_interest2 = cv.bitwise_or(255 - binarized2[y:y+h, x:x+w], rect_image)

            if(255 in region_of_interest):
                #Aqui se filtra la proporcion, mayor height que width
                aspect_ratio = float(w)/h
                if(aspect_ratio < 1): 
                    #La media de los pixeles de la region de interes debe de ser mayor a 0.7, debido a que todo el caracter debe de ocupar el rectangulo
                    if(np.mean(region_of_interest) > 0.7):
                        
                        area = cv.contourArea(contours[c])
                        rect_area = w*h
                        extent = float(area)/rect_area
                        #Se busca que la extension sea mayor a 0.2, es decir que sea un cuadrado o lo más parecido. 
                        if(extent > 0.2):
                            #Se le genera un pad a la imagen a predecir.
                            region_of_interest2 = np.pad(region_of_interest2, 9 , 'constant', constant_values=255)
                            nuevo_tamano = (75, 100)  
                            imagen_redimensionada = cv.resize(region_of_interest2, nuevo_tamano)
                            imagen_redimensionada = imagen_redimensionada.reshape(1, -1)
                            
                            #Se dibujan los cuadrados verdes en los caracteres encontrados
                            if(bandera_siguiente == 1):
                                cv.rectangle(imagen_final, (x1 +x,  y1 + y), (x1 +x+w, y1 +y+h), (0, 255, 0), 2)
                            else:
                                cv.rectangle(imagen_final, (x, y), (x+w,y+h), (0, 255, 0), 2)
                            found = False

                            #Se ordena en base a la poiscion de y
                            for altura_existente, caracteres in contornos_dict.items():
                                if abs(y - altura_existente) < 5:
                                    caracteres.append([x, imagen_redimensionada, ])
                                    found = True
                                    break
                            if not found:
                                # Crea una nueva entrada en el diccionario para la nueva fila
                                contornos_dict[y] = [[x, imagen_redimensionada]]

    #Se ordena en base a la posicion de x     
    contornos_dict = dict(sorted(contornos_dict.items()))
    for clave, lista in contornos_dict.items():
        contornos_dict[clave] = sorted(lista, key=lambda x: x[0])

    #Se genera una sola lista.
    text = []
    for key,value in contornos_dict.items():
        for imagenes in value:
            text.append(imagenes[1])

    #Se realiza la prediccion de los caracteres
    if(len(text) > 0):
        text = np.vstack(text)
        predicciones = model.predict(text)

        resultados = ""
        for pred in predicciones:
            car = number2char(pred)
            resultados = resultados + str(car)
        cv.putText(imagen_final,str(resultados),(20,30),1,2.2,(0,255,0),3)
    else:
        resultados = ""
    return (imagen_final, resultados)




if(len(sys.argv) >= 2):
    bandera = sys.argv[1]
    if(bandera == "--p"):
        img_path = sys.argv[2]

        model = load('./modelo_imagen_procesada.joblib')
        img, gray = leer_imagenes(img_path)

        #Primer Tresh
        binarized2 = binarizacion(gray, 127)
        contours, hierarchy = encontrar_contornos(binarized2)
        bandera_siguiente,x1,y1,w1,h1,placa = encontrar_placa(binarized2, contours, hierarchy)
        if(bandera_siguiente == 1):
            binarized2 = placa
            contours, hierarchy = encontrar_contornos(placa)
        imagen_final1, text1 = encontrar_caracteres(img, binarized2, contours, hierarchy, bandera_siguiente, x1,y1, model)

        #Segundo Tresh
        binarized2 = binarizacion(gray, 127)
        contours, hierarchy = encontrar_contornos(binarized2)
        bandera_siguiente,x1,y1,w1,h1,placa = encontrar_placa(binarized2, contours, hierarchy)
        if(bandera_siguiente == 1):
            binarized2 = placa
            contours, hierarchy = encontrar_contornos(placa)
        imagen_final2, text2 = encontrar_caracteres(img, binarized2, contours, hierarchy, bandera_siguiente, x1,y1, model)

        if(len(text1) > len(text2)):
            print(text1)
            cvlib.imgview(imagen_final1)
        elif(len(text2) > len(text1)):
            print(text2)
            print("yes")
            cvlib.imgview(imagen_final2)
        else:
            print(text1)
            cvlib.imgview(imagen_final1)

    else:
        print("Debe de ingresar todos los parametros: detector.py --p ./imagenes/images108.jpg")
else:
        print("Debe de ingresar todos los parametros: detector.py --p ./imagenes/images108.jpg")
