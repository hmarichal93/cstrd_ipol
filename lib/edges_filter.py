#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:26:28 2022

@author: henry
"""

import numpy as np
import matplotlib.pyplot as plt

import lib.chain_v4 as ch
from lib.io import write_json,load_json, load_data, save_dots, Nr, get_path, ANGLE_THRESHOLD, pliegoGrados

def extraerPixelesPertenecientesAlPerfil(copia, angle, centro=None):
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
    y_pix = []
    x_pix = []
    angle_rad = angle * np.pi / 180
    ctrl = True
    while ctrl:
        x = centro[1] + i * np.sin(angle_rad)
        y = centro[0] + i * np.cos(angle_rad)
        x = x.astype(int)
        y = y.astype(int)

        # print(f'y={y} x={x}')

        if i == 0 or not (x == x_pix[-1] and y == y_pix[-1]):
            y_pix.append(y)
            x_pix.append(x)
        if y >= M - 1 or y <= 1 or x >= N - 1 or x <= 1:
            ctrl = False

        i += 1

    return np.array(y_pix), np.array(x_pix)

def angle_2_coord(angle):
    if angle >= 337.5 or angle < 22.5:
        pos = (-1,0)
        
    elif 22.5<= angle < 67.5:
        pos = (-1,-1) 
    
    elif 67.5 <= angle < 112.5:
        pos = (0,-1)
        
    elif 112.5 <= angle < 157.5:
        pos = (1,-1)
    
    elif 157.5 <= angle < 202.5:
        pos = (1,0)
    
    elif 202.5 <= angle < 247.5:
        pos = (1,1)
    
    elif 245.5 <= angle < 292.5:
        pos = (0,1)
    
    elif 292.5 <= angle < 337.5:
        pos = (-1,1)
    
    return pos

def plotVector(origin, angle):
    
    V,U = angle_2_coord(angle)
    Y, X = origin[0], origin[1]
    # print(f" X {X.shape} Y {Y.shape} U {U.shape} V {V.shape}")
    plt.quiver(
        X,
        Y,
        U,
        V,
        color="k",
        scale=5,
        scale_units="inches",
        angles="xy",
        headwidth=1,
        headlength=3,
        width=0.005,
    )
    # plt.quiver(*origin,vx,vy,angles='xy', scale_units='xy',units='inches')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return (
        vector / np.linalg.norm(vector)
        if np.linalg.norm(vector) > 0
        else np.zeros(vector.shape[0])
    )


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def dict2matriz(puntos, centro):
    matriz = np.array([[-1, -1, -1, -1, -1]])

    for angle in puntos:
        for dot in puntos[angle]:
            radio = np.sqrt((dot[0] - centro[1]) ** 2 + (dot[1] - centro[0]) ** 2)
            row = np.array([dot[0], dot[1], angle, radio, dot[2]], dtype=float)
            matriz = np.vstack((matriz, row))

    matriz = np.delete(matriz, 0, axis=0)

    return matriz


# def angle_between_radial_direction_and_sobel_phase(i, j, centro, phase):
#     radial_angle = (ch.getAngleFromCoordinates(i,j, centro[::-1])+90) % 360
#     phase_shifted = phase # (phase + np.pi) % 2*np.pi
#     constanteRadianes2grados = 180 / np.pi
#     vector_1 = np.array(
#         [
#             np.sin(radial_angle / constanteRadianes2grados),
#             np.cos(radial_angle / constanteRadianes2grados),
#         ]
#     )
#     vector_2 = np.array([np.sin(phase_shifted), np.cos(phase_shifted )])
#     alpha = angle_between(vector_1, vector_2) * constanteRadianes2grados
#     return alpha 


def angle_between_radial_direction_and_sobel_phase(i,j,centro,phase):
    angulo_perfil = ch.getAngleFromCoordinates(i, j, centro[::-1])
    constanteRadianes2grados = 180 / np.pi
    anguloGrad = phase 
    vector_1 = np.array(
        [
            np.sin(angulo_perfil / constanteRadianes2grados),
            np.cos(angulo_perfil / constanteRadianes2grados),
        ]
    )
    vector_2 = np.array([np.sin(anguloGrad), np.cos(anguloGrad )])
    alpha = angle_between(vector_1, vector_2) * constanteRadianes2grados
    return alpha
 
def anguloEntreBordesPerfilYgradiente(faseCandidatosBordes, y,x,centro, debug=False):
    alpha = np.zeros_like(faseCandidatosBordes)

    for index, fasePunto in enumerate(faseCandidatosBordes):
        i,j = y[index],x[index]
        alpha[index] = angle_between_radial_direction_and_sobel_phase(i,j, centro,fasePunto)


    if debug:
        plt.figure()
        plt.plot(alpha, "r")
        plt.plot(angulo_perfil * np.ones(faseCandidatosBordes.shape[0]), "b")
        plt.plot(faseCandidatosBordes * constanteRadianes2grados)

    return alpha


def limpiarBordesRadialesConsecutivos(peaks):
    # Posprocesamiento. Para no tener puntos "dobles" por direccion radial.
    ind2del = []
    umbral = 5  # pixeles
    dobles = np.zeros_like(peaks)
    dobles[1:] = np.where((peaks[1:] - peaks[:-1]) < umbral, 1, 0)
    ind2del = np.where(dobles == 1)[0]
    peaks_del = np.delete(peaks, ind2del) if ind2del.shape[0] > 0 else peaks
    return peaks_del


def bordesColinealesAlPerfil(alpha, candidatos, debug=False, th=ANGLE_THRESHOLD):
    bordes = candidatos[np.where(np.abs(alpha) <=th)[0]]
    postBordes = limpiarBordesRadialesConsecutivos(bordes)
    if debug:
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        axs[0].vlines(bordes, ymin=0, ymax=255, colors="r")
        axs[1].vlines(postBordes, ymin=-np.pi, ymax=np.pi, colors="r")

    return postBordes


def faseGradienteCandidatosBorde(perfil_bin, perfil_fase):
    candidatos_bordes_indices = np.where(perfil_bin == 255)[0]
    faseCandidatosBordes = perfil_fase[candidatos_bordes_indices]
    return faseCandidatosBordes, candidatos_bordes_indices

def build_radial_directions_matrix(img_seg,centro):
    perfils_matrix = np.zeros_like(img_seg)
    rango = np.arange(0, 360, 360 / Nr)
    perfiles = {}
    for angulo_perfil in rango:
        y, x = extraerPixelesPertenecientesAlPerfil(
            img_seg, angulo_perfil, [centro[1], centro[0]]
        )
        
        yy, xx = (
            y.reshape(-1, 1),
            x.reshape(-1, 1)
        )
        perfiles[angulo_perfil] = np.hstack((yy, xx))
        perfils_matrix[yy,xx] = 1 + angulo_perfil
        
    return perfils_matrix
def empaquetadoDePuntosBordesPerfil(bordesPerfil, y, x, perfil_fase):
    yy, xx, fase = (
        y[bordesPerfil].reshape(-1, 1),
        x[bordesPerfil].reshape(-1, 1),
        perfil_fase[bordesPerfil].reshape(-1, 1),
    )
    # if medula:
    #     radios = np.sqrt(np.sum((xx-centro[0])**2+(yy-centro[1])**2,axis=1))
    #     yy = np.delete(yy,np.where(radios<medula)[0]).reshape((-1,1))
    #     xx = np.delete(xx,np.where(radios<medula)[0]).reshape((-1,1))
    #     fase = np.delete(fase,np.where(radios<medula)[0]).reshape((-1,1))

    return np.hstack((yy, xx, fase))


def extraccionPerfil(y, x, img_seg, img_bin_canny, gradFase, debug=False):
    perfil_orig, perfil_bin, perfil_fase = (
        img_seg[y, x],
        img_bin_canny[y, x],
        gradFase[y, x],
    )
    if debug:
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        axs[0].plot(perfil_orig)
        axs[0].grid(True)
        axs[0].set_title("Perfil")
        axs[0].vlines(np.where(perfil_bin == 255)[0], ymin=0, ymax=255, colors="r")

        axs[1].plot(perfil_fase)
        axs[1].set_title("fase")
        axs[1].vlines(
            np.where(perfil_bin == 255)[0], ymin=-np.pi, ymax=np.pi, colors="r"
        )
        axs[1].grid(True)

    return perfil_orig, perfil_bin, perfil_fase


def edgePostprocesing(img_seg, img_bin_canny, gradFase, centro,debug = True):
    rango = np.arange(0, 360, 360 / Nr)
    perfiles = {}
    for angulo_perfil in rango:
        y, x = extraerPixelesPertenecientesAlPerfil(
            img_seg, angulo_perfil, [centro[1], centro[0]]
        )
        perfil_orig, perfil_bin, perfil_fase = extraccionPerfil(
            y, x, img_seg, img_bin_canny, gradFase, debug=False
        )
        faseCandidatosBordes, candidatos = faseGradienteCandidatosBorde(
            perfil_bin, perfil_fase
        )
        alpha = anguloEntreBordesPerfilYgradiente(
            faseCandidatosBordes, y,x,centro, debug=False
        )
        bordesPerfil = bordesColinealesAlPerfil(alpha, candidatos)
        perfiles[angulo_perfil] = empaquetadoDePuntosBordesPerfil(
            bordesPerfil, y, x, perfil_fase
        )


    return perfiles

def filtrado_bordes_colineales_al_perfil(img_bin_canny_mask,centro,gradFase, debug=False):
    cartesianas = img_bin_canny_mask.copy()
    cartesianas = np.where(cartesianas == 0, 255, 0)
    
    if debug:
        plt.figure(figsize=(15, 15))
        plt.imshow(cartesianas, cmap="gray")
        plt.title("Bordes de Canny")
        plt.show()
    
    yy, xx = np.where(img_bin_canny_mask > 0)        
    for idx in tqdm(range(len(xx))):
        i,j = yy[idx], xx[idx]
        alpha = angle_between_radial_direction_and_sobel_phase(i,j, centro,gradFase[i,j])
        
        if 90 - pliegoGrados <= np.abs(alpha) <= 90 + pliegoGrados:
            cartesianas[i - 1 : i + 2, j - 1 : j + 2] = 255


    if debug:
        plt.figure(figsize=(15, 15))
        plt.imshow(cartesianas, cmap="gray")
        plt.title("Zonas donde se borran bordes")
        plt.show()
    
    return np.where(cartesianas == 0, 255, 0).astype(np.uint8)


def filtrado_de_bordes(datos):
    img_bin_canny,gradFase,img_seg,centro,SAVE_PATH = datos['img_bin_canny'], datos['gradFase'], datos['img_seg'], datos['centro'], datos['save_path'] 
    display = datos['display']
    bin_sin_pliegos = filtrado_bordes_colineales_al_perfil(img_bin_canny,centro,gradFase)
    datos['bin_sin_pliegos'] = bin_sin_pliegos
    plt.figure(figsize=(15,15))
    plt.imshow(bin_sin_pliegos,cmap='gray')
    plt.title('bin_sin_pliegos')
    plt.axis('off')
    plt.savefig(f"{SAVE_PATH}/bordes_sin_pliegos.png")
    if display:
        plt.show()
    plt.close()
    perfiles = edgePostprocesing(img_seg, bin_sin_pliegos, gradFase, centro)

    
    datos['perfiles'] = perfiles
    datos['gradFase'] = gradFase
    datos['perfils_matrix'] = build_radial_directions_matrix(img_seg,centro)
    
    return 0
if __name__ == '__main__':
    min_angle = 0
    max_angle = 360
    M = 1000
    N = 500
    img = np.zeros((M,N))
    rango = np.arange(min_angle, max_angle, (max_angle - min_angle) / Nr)
    perfiles = {}
    centro = [int(M/2), int(N/2)]
    for angulo_perfil in rango:
        y, x = extraerPixelesPertenecientesAlPerfil(
            img, angulo_perfil, [centro[0], centro[1]]
        )
        print(y.shape)
        i, j = y[1], x[1]
        print(f" angulo mas cercano {ch.getAngleFromCoordinates(i,j,centro)} angulo_perfil {angulo_perfil}")

        i, j = y[20], x[20]
        print(f" angulo punto medio {ch.getAngleFromCoordinates(i, j, centro)} angulo_perfil {angulo_perfil}")

        i, j = y[-1], x[-1]
        print(f" angulo mas lejano {ch.getAngleFromCoordinates(i,j,centro)} angulo_perfil {angulo_perfil}")
