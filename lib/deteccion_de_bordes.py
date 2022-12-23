#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:44:06 2022

@author: henry
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity, equalize_adapthist
import cv2

import lib.devernayEdgeDetector as dev
from lib import utils
import lib.chain_v4 as ch
import lib.edges_filter as edges


def edge_bin_devernay(nombre, image, sigma, centro, save_path, debug=False):
    detector = dev.devernayEdgeDetector(nombre, image, sigma=sigma, centro=centro, save_path=save_path)
    img_bin, bin_sin_pliegos, thetaMat, nonMaxImg, gradientMat, Gx, Gy, img_labels, listaCadenas, listaPuntos, lista_curvas = detector.detect()

    return img_bin, bin_sin_pliegos, thetaMat, nonMaxImg, gradientMat, Gx, Gy, img_labels, listaCadenas, listaPuntos, lista_curvas


def quitarMedula(img_bin_canny, centro, radio, debug=False):
    height, width = img_bin_canny.shape
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, centro, radio, 255, thickness=-1)
    img_bin_canny[circle_img == 255] = 255
    img_bin_canny_mask = cv2.bitwise_xor(img_bin_canny, circle_img)
    if debug:
        plt.figure(figsize=(15, 15))
        plt.imshow(img_bin_canny_mask, cmap="gray")
        plt.title(f"Medula {radio}")

    return img_bin_canny_mask


def deteccion_de_bordes_sin_procesamiento(nombre, img, sigma, edges='canny', centro=None, save_path=None):
    imageGray = utils.rgbToluminance(img)
    img_seg = imageGray  # utils.segmentarImagen(imageGray, debug=False)
    img_eq = equalize_adapthist(np.uint8(img_seg), clip_limit=0.03)

    img_bin, bin_sin_pliegos, thetaMat, nonMaxImg, gradientMat, Gx, Gy, img_labels, listaCadenas, listaPuntos, lista_curvas = edge_bin_devernay(
        nombre, img_eq, sigma=sigma, centro=centro, save_path=save_path)
    return img_bin, bin_sin_pliegos, thetaMat, nonMaxImg, gradientMat, Gx, Gy, img_labels, listaCadenas, listaPuntos, lista_curvas


def deteccion_de_bordes(datos, edges='canny'):
    M, N, img, centro, SAVE_PATH, sigma = datos['M'], datos['N'], datos['img'], datos['centro'], datos['save_path'], \
                                          datos['sigma']
    nombre = '10'
    to = time.time()

    img_bin, bin_sin_pliegos, thetaMat, nonMaxImg, gradientMat, Gx, Gy, img_labels, listaCadenas, listaPuntos, lista_curvas = deteccion_de_bordes_sin_procesamiento(
        nombre, img, sigma, edges=edges, centro=centro[::-1], save_path=SAVE_PATH)
    datos['img_bin_canny'] = bin_sin_pliegos

    datos['bin_sin_pliegos'] = bin_sin_pliegos

    datos['perfils_matrix'] = np.zeros_like(img)
    datos['perfiles'] = np.zeros_like(img)
    datos['gradFase'] = thetaMat
    datos['nonMaxSup'] = nonMaxImg
    datos['labels_img'] = img_bin
    datos['modulo'] = gradientMat
    datos['img_seg'] = img_bin
    datos['Gy'] = Gy
    datos['Gx'] = Gx
    datos['listaCadenas'] = listaCadenas
    datos['listaPuntos'] = listaPuntos
    M, N, _ = img.shape
    datos['MatrizEtiquetas'] = ch.buildMatrizEtiquetas(M, N, listaPuntos)
    datos['lista_curvas'] = lista_curvas

    img_aux = np.zeros((M, N, 3))
    img_aux[:, :, 1] = thetaMat
    # cv2.imwrite(f"{SAVE_PATH}/gradFase.png",img_aux)

    tf = time.time()
    datos['tiempo_bordes'] = tf - to
    return 0


################################################################################################
##################################### test 
################################################################################################


def get_circles(center, r, epsilon, width, height):
    img = np.zeros((height, width, 3)) + 255
    a, b = center[0], center[1]

    # map_ = [['.' for x in range(width)] for y in range(height)]

    # draw the circle
    for y in range(height):
        for x in range(width):
            # see if we're close to (x-a)**2 + (y-b)**2 == r**2
            if abs((x - a) ** 2 + (y - b) ** 2 - r ** 2) < epsilon ** 2:
                img[y, x] = 0

    return img


def test_sobel():
    N = 1000
    centro = [int(N / 2), int(N / 2)]
    shape = get_circles(centro, r=N * 0.4, epsilon=N * 0.1, width=N, height=N)
    plt.figure(figsize=(15, 15))
    plt.imshow(shape)

    print(shape[20:30, 65:75])
    return shape, np.array([N, N])


def test_edge_detector(sigma, img, centro, edges_det):
    img_seg, img_bin_canny, gradFase, nonMaxSup, modulo, Gy, Gx = deteccion_de_bordes_sin_procesamiento(img, sigma,
                                                                                                        edges=edges_det)
    plt.figure(figsize=(10, 10))
    plt.imshow(gradFase)
    # 2.0 bordes sin procesar
    plt.figure(figsize=(15, 15))
    plt.imshow(img_bin_canny, cmap='gray')
    plt.title('img_bin_canny')
    plt.axis('off')
    plt.figure(figsize=(15, 15))
    plt.imshow(img, cmap='gray')
    plt.title('img_bin_canny')
    plt.axis('off')
    plt.show()
    M = img_seg.shape[0]
    N = img_seg.shape[1]

    bin_sin_pliegos = edges.filtrado_bordes_colineales_al_perfil(img_bin_canny, centro, gradFase, Gy, Gx)

    plt.figure(figsize=(15, 15))
    plt.imshow(bin_sin_pliegos, cmap='gray')
    plt.title('bin_sin_pliegos')
    plt.axis('off')
    plt.show()

    perfiles = edges.edgePostprocesing(img_seg, bin_sin_pliegos, gradFase, centro, Gy, Gx)
    num_labels, labels_im = cv2.connectedComponents(bin_sin_pliegos, connectivity=8)

    listaPuntos = ch.convertirAlistaPuntos(perfiles, labels_im, centro)
    MatrizEtiquetas = -1 + np.zeros_like(labels_im)
    for dot in listaPuntos:
        MatrizEtiquetas[dot.x, dot.y] = dot.cadenaId

    listaCadenas = ch.asignarCadenas(listaPuntos, centro[::-1], M, N)
    ch.visualizarCadenasSobreDisco(
        listaCadenas, img, "cadenas", labels=False, color='r', display=True
    )


if __name__ == "__main__":
    from lib.utils import cargarImagen

    # img,centro = test_sobel()
    image_file_name = "/media/data/maestria/datasets/artificiales/segmentadas/example_1.tif"
    image = cv2.imread(image_file_name)
    centro = [500, 500]
    sigma = 1.5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # test_edge_detector(img,centro, "canny")
    diff = test_edge_detector(sigma, image, centro, edges_det="canny")
