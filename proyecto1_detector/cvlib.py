import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def imgview(img, title=None, axis=False, figsize= None, filename=None):
    """ Recibe un np array y como resultado muestra la imagen visualmente.
    Args:
        img (numpy array): La matriz de la imagen original.
        title (string): Texto que se desea poner como titulo.
        axis (bool): Es un bool que permite agregar ejes a la imagen.
    Returns:
        None
    """
    #Es para configurar el size de la imagen
    base=  5
    #Es la base (Figura)
    if(figsize != None):
        figura = plt.figure(figsize=(base,base),facecolor='black') #El ultimo parametro es para poner en negro el fondo.
    else:
        figura = plt.figure(figsize=figsize,facecolor='black') #
    #Es el eje (Eje)
    ax = figura.add_subplot(111) 
    #Verificar la imagen es a color o no
    if len(img.shape) == 3: 
        im = ax.imshow(img,extent=None)
    else:
        im = ax.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)
    #Verificar si se desea poner un titulo a la imagen 
    if title != None:
        ax.set_title(title,fontsize=14)
    #Verificar si se desea poner los ejes a la imagen   
    if not axis:
        plt.axis('off')
    else:
        ax.grid(c='w')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w',labelsize=14)
        ax.tick_params(axis='y', colors='w',labelsize=14)
    if filename != None:
        plt.savefig(filename)
    plt.show()

def hist(img, fill = False):
    """ recibe un np array y como resultado muestra la imagen visualmente y su histograma a un costado.
    Args:
        img (numpy array): La matriz de la imagen original.
        fill (bool): Es un bool que permite rellenar el color debajo de la curva.
    Returns:
        None
    """
    #Se cambia de BGR a RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #Es para configurar el size de la imagen
    k = 8
    #Es la base (Figura)
    fig = plt.figure(figsize=(k,k))
    #Es el eje (Eje)
    ax = fig.add_subplot(1,2,1)
     #Si la imagen es a color
    if len(img.shape) == 3:
        im = ax.imshow(img,extent=None)
    else:
        im = ax.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)
    plt.axis('off')

    #Se agregan suplots, para la imagen y el histrograma.
    ax1 = fig.add_subplot(1,2,2)
    #Los colores del histograma
    colors = ['r','g','b']
    for i, color in enumerate(colors):
        #Se obtiene el histograma para ese canal
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        #Se hace el plot del histograma
        ax1.plot(histr, c=color, alpha=0.9)
        x = np.arange(0.0, 256, 1)
        #Si es true, se aplica el color debajo de la curva.
        if fill:
            ax1.fill_between(x, 0, histr.ravel(), alpha=0.2, color=color)
    #Se establecen las configuraciones para el plot del histograma
    ax1.set_xlim([0,256])
    ax1.grid(alpha=0.2)
    ax1.set_facecolor('k')
    ax1.set_title('Histogram')
    ax1.set_xlabel('Pixel value')
    ax1.set_ylabel('Pixel count')
    #El asp sirve para que la imagen y el histograma tengan las mismas dimensiones. 
    asp = np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
    ax1.set_aspect(asp)
    plt.show()

def imgcmp(img1, img2, title = None):
    """ recibe dos np array y muestra las imágenes lado a lado para su comparación.
    Args:
        img1 (numpy array): La matriz de la imagen original 1.
        img2 (numpy array): La matriz de la imagen original 2.
        title (list): Lista de strings. 
    Returns:
        None
    """

    # Mostrar la imagen 1
    #Es para configurar el size de la imagen
    k = 8
     #Es la base (Figura)
    fig = plt.figure(figsize=(k,k))
    #Es el eje (Eje)
    ax = fig.add_subplot(1,2,1)
    #Si la imagen es a color
    if len(img1.shape) == 3:
        im = ax.imshow(img1,extent=None)
    else:
        im = ax.imshow(img1,extent=None,cmap='gray',vmin=0,vmax=255)


    # Mostrar la imagen 2
    #Es el eje (Eje)
    ax1 = fig.add_subplot(1,2,2)
     #Si la imagen es a color
    if len(img2.shape) == 3:
        im = ax1.imshow(img2,extent=None)
    else:
        im = ax1.imshow(img2,extent=None,cmap='gray',vmin=0,vmax=255)


    #El len de la lista debe de ser 2 para poder poner los titulos. 
    if(title != None and len(title) == 2):
        ax.set_title(title[0])
        ax1.set_title(title[1])

    plt.show()

def colorhist(img):
    fig, ax = plt.subplots(figsize=(20,8))
    colors = ['r','g','b']
    for i, color in enumerate(colors):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        ax.plot(histr, c=color, alpha=0.9)
        x = np.arange(0.0, 256, 1)
    ax.set_xlim([0,256])
    ax.grid(alpha=0.2)
    ax.set_facecolor('k')
    ax.set_title('Histogram', fontsize=20)
    ax.set_xlabel('Pixel value', fontsize=20)
    ax.set_ylabel('Pixel count', fontsize=20)

    plt.show()






