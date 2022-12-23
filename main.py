#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:54:53 2021

@author: henry
@brief:
    TODO:
        - corrigue superposicion de cadenas azules con rojas.
"""
import time


from lib.deteccion_de_bordes import deteccion_de_bordes
from lib.io import levantar_imagen
import lib.spyder_web_4 as spyder
import lib.chain_v4 as ch
import lib.unir_cadenas_prolijo_3 as union

VERSION = "v3.0.2.3_refinada_performance_centro"

def main(img_name,output_file, sigma, cy, cx):
    to = time.time()
    results = levantar_imagen(img_name, cy, cx, sigma)
    edge_type = 'devernay'
    print("Step 2.0: Detectar bordes")
    deteccion_de_bordes(results, edges=edge_type)
    tf = time.time()
    print(f'Execution Time {tf-to:.1f}')

    print("Step 3.0: Muestreo de bordes")
    to = time.time()
    spyder.main(results)
    tf = time.time()
    print(f'Execution Time {tf-to:.1f}')
    listaCadenas, img = results['listaCadenas'], results['img']
    ch.visualizarCadenasSobreDisco(
        listaCadenas, img, "./chains.png", labels=False, gris=True, color=True
    )
    print("Step 6.0: Unir Cadenas")
    step = "Step6.0"

    union.unir_cadenas(results, step=step)
    listaCadenas, img = results['listaCadenas'], results['img']
    ch.visualizarCadenasSobreDisco(
        listaCadenas, img, "./grouping_chains.png", labels=False, gris=True
    )
    # save_results(results)

    # save_system_status(results)
    # results = load_system_status(nro_imagen, False, VERSION, VERSION)
    tf = time.time()
    union.postprocesamiento_etapa_2(results, step)
    results['tiempo_post'] = time.time() - tf
    tiempo_post = results['tiempo_post']
    print(f'postproceso: {tiempo_post:0.1f}')

    ##save results
    listaCadenas, img = results['listaCadenas'], results['img']
    ch.visualizarCadenasSobreDisco(
        listaCadenas, img, output_file, labels=False, gris=True
    )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(args.input, args.output, args.sigma, args.cy, args.cx)


