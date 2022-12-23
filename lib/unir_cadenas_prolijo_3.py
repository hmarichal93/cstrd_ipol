#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:54:43 2022

@author: henry
"""
import numpy as np
import time
from pathlib import Path
import cv2
from shapely.geometry import LineString, Point, Polygon
import glob
from natsort import natsorted
import matplotlib.pyplot as plt


#propias
from lib.io import Nr
import lib.chain_v4 as ch
import lib.union_puntos_prolijo_performance_3 as union
import lib.union_cadenas_postprocesamiento as union_post
def check_chains_size(lista_cadenas):
    for cadena in lista_cadenas:
        if cadena.size<2:
            print(cadena)
            raise



def put_text(text, image, org, color=(0, 0, 0), fontScale=1 / 4):
    # font
    font = cv2.FONT_HERSHEY_DUPLEX
    # fontScale

    # Blue color in BGR

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    return image


def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)



def unir_cadenas(datos,step):
    listaCadenas, listaPuntos,  M,N = datos['listaCadenas'], datos['listaPuntos'],datos['M'], datos['N']
    img, SAVE_PATH, gradFase, centro =  datos['img'], datos['save_path'], datos['gradFase'], datos['centro']
    to = time.time()
    debug_imgs = datos['debug']
    max_id_cad_puntos = sorted(listaPuntos,key=lambda  x : x.cadenaId,reverse=True)[0]
    max_id_cad = sorted(listaCadenas, key=lambda x: x.id, reverse=True)[0]
    print(f"lenCadenas {len(listaCadenas)} cad_max_lista_puntos {max_id_cad_puntos.cadenaId} cad_max_lista_cad {max_id_cad.id} ")
    print(
        f"listaPuntos {len(listaPuntos)}  listaCadenas {len(listaCadenas)} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    )

    check_chains_size(listaCadenas)
    #if 'matrizIntersecciones' not in datos.keys():
    Matriz_intersecciones = union.calcular_matriz_intersecciones(listaCadenas, listaPuntos)
    # else:
    #     Matriz_intersecciones = datos['matrizIntersecciones']
    #hay_cambios,listaCadenas,listaPuntos,Matriz_intersecciones,MatrizEtiquetas = union.loop_unir_cadenas()
    for cadena in listaCadenas:
        puntos_otro_id = [punto.cadenaId for punto in cadena.lista if punto.cadenaId != cadena.id ]
        if len(puntos_otro_id) > 0:
            raise
    logs_1 = Path(SAVE_PATH) / "debug_1"
    import os
    if logs_1.exists() and debug_imgs:
        os.system(f"rm -rf {str(logs_1)}")
    if debug_imgs:
        logs_1.mkdir(parents=True, exist_ok=True)
    listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas,listaPuntos,Matriz_intersecciones,img,gradFase,
                                             centro, path=str(logs_1), radial_tolerance = 0.1, debug_imgs=debug_imgs,ancho_std=2,
                                                                distancia_angular_maxima=10
                                                                )

    listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas,listaPuntos,Matriz_intersecciones,img,gradFase,
                                             centro, path=str(logs_1), radial_tolerance = 0.2, debug_imgs=debug_imgs,ancho_std=2,
                                                                distancia_angular_maxima=10
                                                                )
    #listaCadenas,listaPuntos = ch.renombrarCadenas(listaCadenas, listaPuntos)
    #MatrizEtiquetas = ch.buildMatrizEtiquetas(M, N, listaPuntos)
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_1',labels=False,save=SAVE_PATH)
    #
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_1_labels',labels=True,save=SAVE_PATH)
    # print(
    #     f"listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    # )
    #% imagenes en gris
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_1_gris',labels=False,save=SAVE_PATH,gris=True)
    #%
    #print("Step 2.6.1: Union Cadenas Criterios Fuertes: Interseccion Con menos restricciones")

    #Matriz_intersecciones = union.calcular_matriz_intersecciones(listaCadenas, debug=False)
    # hay_cambios,listaCadenas,listaPuntos,Matriz_intersecciones,MatrizEtiquetas = union.loop_unir_cadenas()

    ##############################33 control
    for ch_up in listaCadenas:
        assert len([punto for punto in ch_up.lista if punto.cadenaId != ch_up.id]) == 0
    ##################################
    logs_1 = Path(SAVE_PATH) / "debug_2"
    if logs_1.exists() and debug_imgs:
        os.system(f"rm -rf {str(logs_1)}")
    if debug_imgs: logs_1.mkdir(parents=True, exist_ok=True)
    ch.verificacion_complitud(listaCadenas)
    listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img,
                                            gradFase, centro,path=str(logs_1), radial_tolerance= 0.1,  distancia_angular_maxima=22,debug_imgs=debug_imgs,ancho_std=3)

    listaPuntos, listaCadenas, Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img,
                                                                  gradFase, centro, path=str(logs_1),
                                                                  radial_tolerance=0.2, distancia_angular_maxima=22,
                                                                  debug_imgs=debug_imgs, ancho_std=3)

    ch.verificacion_complitud(listaCadenas)
    tf = time.time()
    # listaCadenas,listaPuntos = ch.renombrarCadenas(listaCadenas, listaPuntos)
    # MatrizEtiquetas = ch.buildMatrizEtiquetas(M, N, listaPuntos)
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_2',labels=False,save=SAVE_PATH)
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_2_labels',labels=True,save=SAVE_PATH)
    # print(
    #     f"listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    # )
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_2_gris',labels=False,save=SAVE_PATH,gris=True)



    logs_1 = Path(SAVE_PATH) / "debug_3"
    if logs_1.exists() and debug_imgs:
        os.system(f"rm -rf {str(logs_1)}")
    if debug_imgs: logs_1.mkdir(parents=True, exist_ok=True)

    listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img, gradFase,
                                            centro, path = str(logs_1), radial_tolerance = 0.1, distancia_angular_maxima = 45,
                                                                debug_imgs=debug_imgs, todas_intersectantes=False, ancho_std=3)

    listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img, gradFase,
                                            centro,path=str(logs_1), radial_tolerance= 0.2, distancia_angular_maxima=45,
                                                                debug_imgs=debug_imgs,todas_intersectantes=False,ancho_std=3)

    listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img, gradFase,
                                            centro,path=str(logs_1), radial_tolerance= 0.1, distancia_angular_maxima=22,
                                                                debug_imgs=debug_imgs,todas_intersectantes=False,ancho_std=2, der_desde_centro= True)
    listaPuntos, listaCadenas, Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img,
                                                                  gradFase,
                                                                  centro, path=str(logs_1), radial_tolerance=0.2,
                                                                  distancia_angular_maxima=45,
                                                                  debug_imgs=debug_imgs, todas_intersectantes=False,
                                                                  ancho_std=3, der_desde_centro=True)

    # listaPuntos,listaCadenas,Matriz_intersecciones = union.main(listaCadenas, listaPuntos, Matriz_intersecciones, img, gradFase,
    #                                         centro,path=str(logs_1), radial_tolerance= 0.2, distancia_angular_maxima=90,
    #                                                             debug_imgs=debug_imgs,todas_intersectantes=False,ancho_std=3)
    tf = time.time()
    # MatrizEtiquetas = ch.buildMatrizEtiquetas(M, N, listaPuntos)
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_3',labels=False,save=SAVE_PATH)
    # ch.visualizarCadenasSobreDisco(listaCadenas,img,f'{step}-PegarCadenasInterseccion_3_labels',labels=True,save=SAVE_PATH)
    # print(
    #     f"listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    # )

    #%
    datos['union_tiempo'] = tf-to
    print(tf-to)

    print(f"El proceso de union tomo {datos['union_tiempo']/60:.2f}")
    datos['listaCadenas'] = listaCadenas
    datos['listaPuntos'] = listaPuntos
    #datos['MatrizEtiquetas'] = MatrizEtiquetas

    
    return 0






def generate_pdf(path):
    pdf = FPDF()
    pdf.set_font('Arial', 'B', 16)

    figures = glob.glob(f"{path}/**/*.png", recursive=True)
    print(figures)
    for fig in tqdm(natsorted(figures)):
        x, y = 0, 50
        height = 150
        width = 180

        pdf.add_page()
        pdf.image(fig, x, y, h=height, w=width)

    pdf.output(f"{path}/debuggin_pdf.pdf", "F")


def construir_poligono_limite(anillo_externo, anillo_interno):
    if anillo_externo is not None and anillo_interno is not None:
        x, y = anillo_externo.exterior.coords.xy
        pts_ext = [[j, i] for i, j in zip(y, x)]
        x, y = anillo_interno.exterior.coords.xy
        pts_int = [[j, i] for i, j in zip(y, x)]
        poligono = Polygon(pts_ext, [pts_int])

    else:
        if anillo_externo is None:
            x, y = anillo_interno.exterior.coords.xy
        else:
            x, y = anillo_externo.exterior.coords.xy
        pts_ext = [[j, i] for i, j in zip(y, x)]
        poligono = Polygon(pts_ext)

    return poligono

def buscar_cadenas_interiores_shapely(cadenas_incompletas_shapely, anillo_externo, anillo_interno):
    poligono = construir_poligono_limite( anillo_externo, anillo_interno)
    contains = np.vectorize(lambda p: poligono.contains(Point(p)), signature='(n)->()')
    subconjunto_cadenas_interiores_shapely = []
    for cadena in cadenas_incompletas_shapely:
        x, y = cadena.xy
        pts = [[i, j] for i, j in zip(y, x)]
        if len(pts)==0:
            continue
        try:
            vector = contains(np.array(pts))
        except Exception as e:
            continue
        if anillo_externo is not None:
            if vector.sum() == vector.shape[0]:
                subconjunto_cadenas_interiores_shapely.append(cadena)
        else:
            if vector.sum() == 0:
                subconjunto_cadenas_interiores_shapely.append(cadena)

    return subconjunto_cadenas_interiores_shapely


def postprocesamiento_etapa_2(datos,step):

    listaCadenas, listaPuntos = datos['listaCadenas'], datos['listaPuntos']
    listaCadenas,listaPuntos, MatrizEtiquetas = ch.renombrarCadenas(listaCadenas,listaPuntos, datos['M'], datos['N'])
    datos['listaCadenas'], datos['listaPuntos'] = listaCadenas,listaPuntos
    SAVE_PATH, img = datos['save_path'], datos['img']
    #ch.visualizarCadenasSobreDisco(listaCadenas, img, f'{step}-PegarCadenasInterseccion_4_labels_init', labels=True,
    #                               save=SAVE_PATH)


    union_post.main_postprocesamiento(datos, debug=False)

    listaCadenas, listaPuntos, M, N = datos['listaCadenas'], datos['listaPuntos'], datos['M'], datos['N']
    img, SAVE_PATH, gradFase, centro = datos['img'], datos['save_path'], datos['gradFase'], datos['centro']
    # ch.visualizarCadenasSobreDisco(listaCadenas, img, f'{step}-PegarCadenasInterseccion_4', labels=False,
    #                                save=SAVE_PATH)
    # ch.visualizarCadenasSobreDisco(listaCadenas, img, f'{step}-PegarCadenasInterseccion_4_labels', labels=True,
    #                                save=SAVE_PATH)
    #
    ch.visualizarCadenasSobreDisco(listaCadenas, img, f'{step}-PegarCadenasInterseccion_4_gris', labels=False,
                                   save=SAVE_PATH, gris=True)
def convertir_cadena_shapely_a_cadena(cadenas_incompletas_shapely, cadenas_incompletas, subconjunto_cadenas_interiores_shapely):
    conjunto_cadenas_interiores = [cadenas_incompletas[cadenas_incompletas_shapely.index(cad_shapely)]
                                   for cad_shapely in subconjunto_cadenas_interiores_shapely]
    conjunto_cadenas_interiores.sort(key=lambda x: x.size, reverse=True)
    return conjunto_cadenas_interiores


def test_union_cadenas(res):
    listaPuntos, listaCadenas, MatrizEtiquetas, gradFase, M, N = res['listaPuntos'], res['listaCadenas'], res['MatrizEtiquetas'], res['gradFase'], res['M'], res['N']
    #ch.checkListas(listaCadenas, listaPuntos)
    ch.checkMatrizEtiquetasCadenas(listaCadenas, MatrizEtiquetas)
    print(
        f"listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    )
    
    
    listaCadenas, listaPuntos, MatrizEtiquetas = ch.renombrarCadenas(listaCadenas, listaPuntos, M,N)
    
    ch.checkMatrizEtiquetasCadenas(listaCadenas, MatrizEtiquetas)
    print(
        f"listaPuntos {len(listaPuntos)} MatrizEtiquetas {np.where(MatrizEtiquetas>-1)[0].shape} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    )
    
    res['listaCadenas'] = listaCadenas
    res['listaPuntos'] = listaPuntos
    
    step = 'test'
    unir_cadenas(res, step)
