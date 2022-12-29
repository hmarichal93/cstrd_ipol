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
from pathlib import Path

import lib.devernayEdgeDetector as edge_detector
from lib.io import load_image, get_path, load_json, write_json
import lib.spyder_web_4 as muestreo
import lib.unir_cadenas_prolijo_3 as union
from lib.utils import save_results
import lib.preprocesamiento as preprocesamiento


def main(img_name,output_dir, cy, cx):
    t0 = time.time()
    ####################################################################################################################
    results = load_image(img_name, cy, cx, output_dir)

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
    union.unir_cadenas(results)

    ####################################################################################################################
    print("Step 5.0: Post procesamiento")
    union.postprocesamiento_etapa_2(results)

    ####################################################################################################################
    print("Step 6.0: Saving Results")
    save_results(results, results['save_path']/"output.png")

    ####################################################################################################################
    tf = time.time()
    print(f'Total exec time: {tf-t0:.1f} seconds')

    return results

def save_config(args, root_path):
    config = load_json(Path(root_path) / './config/default.json')
    if args.nr:
        config['Nr'] = args.nr

    if args.hsize and args.wsize:
        config['resize'] = [args.hsize, args.wsize]

    if args.min_lenght:
        config["min_chain_lenght"] = args.min_lenght

    if args.edge_th:
        config["edge_th"] = args.edge_th

    if args.sigma:
        config['sigma'] = args.sigma

    if args.th_high:
        config['th_high'] = args.th_high

    if args.th_low:
        config['th_low'] = args.th_low

    write_json(config,get_path('config') / 'general.json')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--root", type=str, required=True)

    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--nr", type=int, required=False)
    parser.add_argument("--hsize", type=int, required=False)
    parser.add_argument("--wsize", type=int, required=False)
    parser.add_argument("--min_lenght", type=int, required=False)
    parser.add_argument("--edge_th", type=int, required=False)
    parser.add_argument("--th_high", type=int, required=False)
    parser.add_argument("--th_low", type=int, required=False)

    args = parser.parse_args()
    save_config(args, args.root)

    main(args.input, args.root, args.cy, args.cx)


