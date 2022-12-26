import os
import json
import socket
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt

import lib.chain_v4 as ch

Nr = 360
ANGLE_THRESHOLD = 45
pliegoGrados = 60
UMBRAL_PEGAR = 0.1


def img_show(img, centro, save=None, titulo=None, color="gray", debug=False):
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
    if debug:
        plt.show()
    plt.close()
    return fig


def cargarImagen(filename):
    # CARGAR IMAGENES DE INTERES
    imagen_original = imageio.imread(filename)

    return imagen_original


def levantar_imagen(image_name, cy, cx, sigma):
    centro = ( cy, cx)

    results = {}

    img = cargarImagen(image_name)

    M, N, _ = img.shape

    results['img'] = img

    results['centro'] = centro
    results['save_path'] = get_path('results')
    results['sigma'] = sigma
    results['M'] = M
    results['debug'] = False

    results['N'] = N

    return results


def get_files(path, ext):
    return glob.glob(f"{path}/**/*.{ext}", recursive=True)


def save_dots(listaPuntos, centro, M, N, nonMaxSup, gradFase, filename):
    isExist = os.path.exists(filename)
    if not isExist:
        os.makedirs(filename)
    result = {}
    lista = []
    for dot in listaPuntos:
        fila = [int(dot.x), int(dot.y), float(dot.angulo), float(dot.radio), float(dot.gradFase), int(dot.cadenaId)]
        lista.append(fila)

    # print(non_max_sup)
    result['puntos'] = lista
    result['centro'] = centro
    result['M'] = M
    result['N'] = N
    print(filename)
    write_json(result, f"{filename}/res.json")
    np.savetxt(f"{filename}/gradFase.csv", gradFase)
    np.savetxt(f"{filename}/nonMaxSup.csv", nonMaxSup)


def load_data(filename):
    result_load = load_json(f"{filename}/res.json")

    lista_puntos = []

    for fila in result_load['puntos']:
        params = {
            "x": fila[0],
            "y": fila[1],
            "angulo": fila[2],
            "radio": fila[3],
            "gradFase": fila[4],
            "cadenaId": fila[5],
        }

        punto = ch.Punto(**params)

        lista_puntos.append(punto)

    centro = result_load['centro']
    M = result_load['M']
    N = result_load['N']

    lista_cadenas = ch.asignarCadenas(lista_puntos, centro[::-1], M, N)

    nonMaxSup = np.genfromtxt(f"{filename}/nonMaxSup.csv")
    gradFase = np.genfromtxt(f"{filename}/gradFase.csv")
    return lista_cadenas, lista_puntos, nonMaxSup, gradFase, M, N


def load_txt(pred_file):
    width = 1920
    height = 1080
    file = open(pred_file, 'r')
    pred_lines = file.readlines()
    # print(pred_lines)
    file.close()
    content = []

    for line in pred_lines:
        line.replace('\n', '')
        # print(line)
        obj, xc, yc, w, h, conf = line.split(' ')
        content.append(
            [int(obj), float(xc) / width, float(yc) / height, float(w) / width, float(h) / height, float(conf)])
    content = np.array(content)
    return content


def load_json(filepath: str) -> dict:
    """
    Load json utility.
    :param filepath: file to json file
    :return: the loaded json as a dictionary
    """
    with open(str(filepath), 'r') as f:
        data = json.load(f)
    return data


def write_json(dict_to_save: dict, filepath: str) -> None:
    """
    Write dictionary to disk
    :param dict_to_save: serializable dictionary to save
    :param filepath: path where to save
    :return: void
    """
    with open(str(filepath), 'w') as f:
        json.dump(dict_to_save, f)


def get_path(*args):
    """
    Return the path of the requested dir/s
    Possible arguments: "data", "bader_data", "training", "results"
    :return: Path/s
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    paths = load_json(f"{dir_path}/../paths_config.json")
    hostname = 'henry-workstation'
    assert hostname in paths.keys(), "Current host: {}, Possible hosts: {}".format(hostname, paths.keys())
    assert all([arg in paths[hostname].keys() for arg in args]), "Args must be in {}".format(paths[hostname].keys())
    paths = tuple([Path(paths[hostname][arg]) for arg in args])
    return paths[0] if len(paths) == 1 else paths


def list_to_matrix(lista_puntos):
    lista = []
    for dot in lista_puntos:
        fila = [dot.x, dot.y, float(dot.angulo), float(dot.radio), float(dot.gradFase), int(dot.cadenaId)]
        lista.append(fila)

    return np.array(lista)


def from_matrix_to_list_dots(matrix):
    listaPuntos = []
    lenght_dots = matrix.shape[0]
    for idx in range(lenght_dots):
        j, i, angulo, radio, fase, cad_id = matrix[idx, 0], matrix[idx, 1], matrix[idx, 2], matrix[idx, 3], matrix[
            idx, 4], matrix[idx, 5]
        params = {'x': j, 'y': i, 'angulo': angulo, 'radio': radio, 'gradFase': fase, 'cadenaId': int(cad_id)}
        punto = ch.Punto(**params)
        # if punto not in listaPuntos:
        listaPuntos.append(punto)
    return listaPuntos


def load_system_status(nroImagen, display, version_salvar, version_cargar):
    results = levantar_imagen(nroImagen, display, version_cargar, version_salvar, debug=True)
    save_path = results['save_path']
    load_path = results['load_path']
    gradFase = np.genfromtxt(f"{load_path}/gradFase.csv")
    matrix = np.genfromtxt(f"{load_path}/matrix.csv")
    modulo = np.genfromtxt(f"{load_path}/modulo.csv")
    nonMaxSup = np.genfromtxt(f"{load_path}/nonMaxSup.csv")
    # perfils_matrix = np.genfromtxt(f"{load_path}/perfils.csv")
    MatrizIntersecciones = np.genfromtxt(f"{load_path}/intersecciones.csv")
    results['gradFase'] = gradFase
    results['modulo'] = modulo
    results['nonMaxSup'] = nonMaxSup
    # results['perfils_matrix'] = perfils_matrix
    # generar lista_puntos
    results['listaPuntos'] = from_matrix_to_list_dots(matrix)
    # generar lista_cadenas
    listaPuntos, centro, M, N = results['listaPuntos'], results['centro'], results['M'], results['N']
    results['listaCadenas'] = ch.asignarCadenas(listaPuntos, centro[::-1], M, N)
    ch.verificacion_complitud(results['listaCadenas'])
    results['MatrizEtiquetas'] = ch.buildMatrizEtiquetas(M, N, listaPuntos)
    results['matrizIntersecciones'] = MatrizIntersecciones

    return results


def save_system_status(results):
    save_path, lista_puntos, gradFase, nonMaxSup, modulo = results['save_path'], results['listaPuntos'], results[
        'gradFase'], \
                                                           results['nonMaxSup'], results['modulo']
    listaCadenas = results['listaCadenas']
    ch.verificacion_complitud(listaCadenas)
    print(
        f"listaPuntos {len(lista_puntos)} cadenas {ch.contarPuntosListaCadenas(listaCadenas)}"
    )
    intersecciones = results['matrizIntersecciones']
    # perfils_matrix = results['perfils_matrix']
    matrix = list_to_matrix(lista_puntos)
    np.savetxt(f"{save_path}/gradFase.csv", gradFase)
    np.savetxt(f"{save_path}/nonMaxSup.csv", nonMaxSup)
    np.savetxt(f"{save_path}/modulo.csv", modulo)
    np.savetxt(f"{save_path}/matrix.csv", matrix)
    # np.savetxt(f"{save_path}/perfils.csv",perfils_matrix)
    np.savetxt(f"{save_path}/intersecciones.csv", intersecciones)


