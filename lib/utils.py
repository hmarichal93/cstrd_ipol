#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:01:24 2021

@author: henry
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio

import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from scipy import stats
from lib.io import Nr
from skimage.transform import AffineTransform, warp
from skimage.transform import rotate

#segmentar
from skimage.filters import median
from skimage.morphology import disk
from scipy import ndimage
from scipy import ndimage as ndi
from skimage.feature import canny
import pandas as pd
from PIL import Image
import re
import logging 
import cv2


from lib.io import write_json


base='FOTOS_DISCOS_1/'
base='FOTOS_DISCOS_2_MEDIDAS_ANILLOS_Y_PRESENCIAMC/'

import lib.chain_v4 as ch

def get_chains_within_angle(angle, lista_cadenas):
    chains_list = []
    for chain in lista_cadenas:
        A = chain.extA.angulo
        B = chain.extB.angulo
        if ((A <= B and A <= angle <= B) or
                (A > B and (A <= angle or angle <= B))):
            chains_list.append(chain)

    return chains_list


def chain_2_labelme_json(chain_list,image_path,image_height,image_width):
    """
    lista_cadenas: lista de cadenas completas
    """
    cadenas_completas = [cadena for cadena in chain_list if cadena.esta_completa()]

    angulo_inicial = 180
    direccion = angulo_inicial
    cadenas_completas_180 = get_chains_within_angle( direccion, cadenas_completas)
    #assert len(cadenas_completas) == len(cadenas_completas_180)



    #1.0
    # punto_cadenas = [ ch.get_closest_chain_dot_to_angle( cadena, direccion) for cadena in cadenas_completas]
    # punto_cadenas.sort(key = lambda x: x.radio)
    # cadenas_completas_ordenadas = [ ch.getChain( dot.cadenaId, cadenas_completas) for dot in punto_cadenas]
    cadenas_completas_ordenadas = cadenas_completas
    labelme_json = {"imagePath":image_path, "imageHeight":image_height,
                    "imageWidth":image_width, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData":None}

    for idx, cadena in enumerate(cadenas_completas_ordenadas):
        anillo = {"label":str(idx+1)}
        y,x = cadena.getDotsCoordinates()
        anillo["points"] = [[i,j] for i,j in np.vstack((x,y)).T.astype(list)]
        anillo["shape_type"]="polygon"
        anillo["flags"]={}
        labelme_json["shapes"].append(anillo)
    return labelme_json

def save_results(datos,output_file):
    listaCadenas= datos['listaCadenas']
    SAVE_PATH= datos['save_path']
    #df_radial, df_general = datos['df_radial'], datos['df_general']
    #% reporte union


    M = datos.get('M')
    N = datos.get('N')
    image_path = datos.get("image_path")
    labelme_json = chain_2_labelme_json(listaCadenas, image_path, M, N)
    write_json(labelme_json, filepath=f"{SAVE_PATH}/labelme.json")

    listaCadenas, img = datos['listaCadenas'], datos['img']
    ch.visualizarCadenasSobreDisco(
        [cad for cad in listaCadenas if cad.esta_completa() and not cad.is_center], img, output_file, labels=False, gris=True
    )

    return 0

    
import os
from pathlib import Path

def setup_log(nroImagen,VERSION):
    logname = f"./logs/{nroImagen}.log"
    if Path(logname).exists():
        os.remove(logname)
    logging.basicConfig(
        filename=logname,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    
    
    logger = logging.getLogger(f"{VERSION}")
    logger.setLevel(logging.INFO)
    logging.info("Init")

    return logger

def write_log(module_name,function,message,level='info',debug=False):
    if debug:
        string = f"[{module_name}][{function}] {message}"
        if level in 'info':
            logging.info(string)
        else:
            logging.error(string)

def img_show(img, centro, save=None,titulo=None, color="gray"):
    fig = plt.figure(figsize=(10, 10))
    if color == "gray":
        plt.imshow(img, cmap=color)
    else:
        plt.imshow(img)
    plt.title(titulo)
    plt.scatter(centro[0], centro[1])
    plt.axis("off")
    if save:
        plt.savefig(f"{save}/original.png")
    return fig

def cargarImagen(filename, centro, debug=False,save=None):
    # CARGAR IMAGENES DE INTERES
    imagen_original = imageio.imread(filename)
    if imagen_original.shape[2] > 3:
        imagen_original = imagen_original[:, :, [0, 1, 2]]
    if debug:
        img_show(imagen_original, centro,save=save)
    return imagen_original

def histograma(I, nBins):
    M,N = I.shape
    hist = np.zeros(nBins)
    for i in range(M):
        for j in range(N):
            hist[int(I[i,j])]+= 1

    hist_n = hist/(M*N)
    
    return hist_n

def histogramaAcumulado(I,nBins):
    hist = histograma(I,nBins)
    cdf = np.zeros(nBins)
    
    cdf[0] = hist[0]
    for i in range(1,nBins):
        cdf[i]= hist[i]+cdf[i-1]

    return cdf
def ecualizarHistograma(I):
    L=256
    M,N = I.shape
    cdf = histogramaAcumulado(I,L)
    feq = np.floor(cdf*(L-1))
    I_eq = feq[I].astype(int)
    return I_eq

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def segmentarImagen(imageGray,debug=False):


    I = median(imageGray,disk(5))
    #image_show(I)
    edges = canny(I/255.)
    #fill_coins = ndi.binary_fill_holes(edges)
    if debug:
        image_show(edges)
    
    
    
    # Elemento estructura1 conectividad 8
    struct2 = ndimage.generate_binary_structure(2,2)
    
    iteraciones=6
    
    dil = ndimage.binary_dilation(edges,structure=struct2,iterations=iteraciones).astype(edges.dtype)
    
    #image_show(dil)
    
    # llenar
    
    fill_coins = ndi.binary_fill_holes(dil)
    #image_show(fill_coins)
    
    
    
    
    iteraciones=30
    
    ero = ndimage.binary_erosion(fill_coins,structure=struct2,iterations=iteraciones).astype(fill_coins.dtype)
    dil2 = ndimage.binary_dilation(ero,structure=struct2,iterations=iteraciones-5).astype(edges.dtype)
    
    #image_show(dil2)
    
    
    tronco = np.where(dil2==True)
    
    img_seg = np.zeros(imageGray.shape)   
    img_seg[tronco] = imageGray[tronco]

    return img_seg

def rgbToluminance(img):
        M,N,C = img.shape
        imageGray = np.zeros((M,N))
        imageGray[:,:] = (img[:,:,0]*0.2126 + img[:,:,1]*0.7152 + img[:,:,2]*0.0722).reshape((M,N))

        return imageGray
from scipy.signal import find_peaks



def smoothingProfile(perfil):
    filtered = []
    for i in range(1,len(perfil)-1):
        filtered.append(np.mean(perfil[i-1:i+2]))
    
    return np.array(filtered)
def moving_average(x, w,windows):
    return np.convolve(x, windows, 'valid') / w

from scipy.ndimage import gaussian_filter

def filterSobel(I):
    from scipy import signal
    
    sobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    Ix = signal.convolve2d(I, sobelX, mode ='same', boundary = 'symm')
    Iy = signal.convolve2d(I, sobelY, mode='same',boundary ='symm')
    
    return Ix,Iy


def buscarBorde(copia,angle,end,radio=0,centro=None):
    """
        angulo =  {0,pi/4,pi/2,3pi/4,pi,5pi/4,6pi/4,7pi/4}
        ptosCard= {S, SE , E  , NE  , N, NW  , W   , SW   }
         | 
        ----------->x
         |
         | IMAGEN
         |
         y
         
    """
    i = 0
    M, N = copia.shape
    y_pix =[]
    x_pix = []
    
    background = 0
        
    ctrl = True      
    while ctrl:
        x = centro[1] + i*np.sin(angle)
        y = centro[0] + i*np.cos(angle)
        x = x.astype(int)
        y = y.astype(int)
      

        if i==0 or not (x==x_pix[-1] and y==y_pix[-1]):
            y_pix.append(y)
            x_pix.append(x)
        #if 
        if end == 'radio':
            r = np.sqrt((x-centro[1])**2 + (y-centro[0])**2)

            if r>radio:
                borde = [y,x]
                ctrl = False
        elif background == copia[y,x]:
                borde = [y,x]
                ctrl = False
        i +=1


    return borde,np.array(y_pix),np.array(x_pix)

def extraerPerfiles(img_seg,centro,debug = False):
    angle = {}
    #oeste
    angle['W'] = -np.pi/2
    #sur
    angle['S'] = 0
    #este
    angle['E'] = np.pi/2
    #norte
    angle['N'] = np.pi
    #noreste
    angle['NE'] = np.pi*3/4
    #noroste
    angle['NW'] = np.pi*5/4
    #suroeste
    angle['SW'] = -np.pi/4
    #sureste
    angle['SE'] = np.pi/4
    
    perfiles = {}
    
    for ptc,angulo in sorted(angle.items()):
        if debug:
            print(f'Punto cardianl {ptc} Angulo {angulo}')
    
        borde,y,x = buscarBorde(img_seg,angulo,end='background',centro=centro)
        perfil = img_seg[y,x]
        perfiles[ptc] = perfil
        #segmento = img_seg[y[::-1],x[0]-100:x[0]+100] 
        if debug: 
            fig, axs = plt.subplots(1)
            fig.suptitle(ptc)
            axs.plot(perfil)
    
    return perfiles



def extraerSubperfiles(perfiles,base,archivo,debug=False):
    """
        extraer los perfiles basados en los radios acumulados medidos manualmente por un experto
    """
    tif = Image.open(base+archivo)
    #image_resolution = 2.54/tif.info['dpi'][0]
    print(tif.info)
    image_resolution = tif.info['dpi'][0]/25.4
    print(tif.info['dpi'])
    print(image_resolution)
    if archivo[0]=='F':
        data = pd.read_csv(base+'fymsa.csv',sep=';')
    else:
        data = pd.read_csv(base+'lumin.csv',sep=';')
    
    
    partido = re.split(r'\.',archivo)
    print(partido)
    dfFoto = data[data['Codigo'] == partido[0]]
    
    radiosReales = {}
    etiquetasTodas = {}
    i=0
    puntoCardinales= ['N','S','E','W']
    
    for i in range(len(puntoCardinales)):
        puntoC = puntoCardinales[i]
        r = dfFoto[f'r{puntoC} mm anual'].values.astype(float)
        etiquetas = dfFoto[f'{puntoC}'].values
        
        r_pix = (r*image_resolution).cumsum().astype(int)
        print(r_pix)
        print(r.cumsum().astype(int))
        perfil = perfiles[puntoC]
    
        if debug:
            plt.figure()
    
        radiosReales[puntoC]=[]
        etiquetasTodas[puntoC] = []
    
        inicio = 0
        for i,fin in enumerate(r_pix):
            if fin<perfil.shape[0]:
                radiosReales[puntoC].append(perfil[inicio:fin])
                etiquetasTodas[puntoC].append(etiquetas[i])
                inicio = fin-1
            
    
    return radiosReales,etiquetasTodas


