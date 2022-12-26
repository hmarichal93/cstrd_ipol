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


import lib.devernayEdgeDetector as edge_detector
from lib.io import levantar_imagen
import lib.spyder_web_4 as muestreo
import lib.chain_v4 as ch
import lib.unir_cadenas_prolijo_3 as union
from lib.utils import save_results
import lib.preprocesamiento as preprocesamiento

VERSION = "v3.0.2.3_refinada_performance_centro"

def main(img_name,output_dir, sigma, cy, cx):
    t0 = time.time()
    ####################################################################################################################
    results = levantar_imagen(img_name, cy, cx, sigma,output_dir)

    ####################################################################################################################
    print("Step 1.0 Preprocessing")
    preprocesamiento.main(results)

    ####################################################################################################################
    print("Step 2.0: Detectar bordes")
    edge_detector.main(results)

    ####################################################################################################################
    print("Step 3.0: Muestreo de bordes")
    muestreo.main(results)

    ####################################################################################################################
    print("Step 4.0: Unir Cadenas")
    #union.unir_cadenas(results, step=step)

    ####################################################################################################################
    print("Step 5.0: Post procesamiento")
    #union.postprocesamiento_etapa_2(results, step)

    ####################################################################################################################
    print("Step 6.0: Saving Results")
    save_results(results, results['save_path']/"output.png")

    ####################################################################################################################
    tf = time.time()
    print(f'Total exec time: {tf-t0:.1f} seconds')

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--outputdir", type=str, required=True)

    args = parser.parse_args()

    main(args.input, args.outputdir, args.sigma, args.cy, args.cx)


