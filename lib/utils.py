#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:01:24 2021

@author: henry
"""
import imageio
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from PIL import Image
import re
import logging 
from typing import List

from lib.io import write_json
import lib.chain as ch


def chain_2_labelme_json(chain_list: List[ch.Chain],image_path,image_height,image_width, img_orig, exec_time):
    """
    chain list: chain list
    """
    init_height, init_width, _ = img_orig.shape
    completed_chains = [chain for chain in chain_list if chain.is_full() and  chain.type not in [ ch.TypeChains.center, ch.TypeChains.border]]


    labelme_json = {"imagePath":image_path, "imageHeight":None,
                    "imageWidth":None, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData": None, 'exec_time(s)':exec_time}
    width_cte = init_width / image_width if image_width is not None else 1
    height_cte = init_height / image_height if image_height is not None else 1
    for idx, chain in enumerate(completed_chains):
        ring = {"label":str(idx+1)}
        ring["points"] = [[node.x*width_cte,node.y*height_cte] for node in chain.nodes_list]
        ring["shape_type"]="polygon"
        ring["flags"]={}
        labelme_json["shapes"].append(ring)

    return labelme_json


def save_results(results,output_file):
    listaCadenas= results['chain_list']
    SAVE_PATH= results['save_path']

    M = results.get('M')
    N = results.get('N')
    image_path = results.get("image_path")
    labelme_json, cadenas_completas = chain_2_labelme_json(listaCadenas, image_path, M, N)
    write_json(labelme_json, filepath=f"{SAVE_PATH}/labelme.json")

    listaCadenas, img = results['chain_list'], results['img']
    ch.visualize_chains_over_image([cad for cad in cadenas_completas if not cad.is_center and not cad.corteza], img,
                                   output_file, labels=False, save=SAVE_PATH, gray=True)
    results['tf'] = time.time()
    print(f"Execution Time: {results['tf']-results['to']}")
    return 0

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





