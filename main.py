#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: henry marichal, hmarichal93@gmail.com
@brief: Method for delineating tree ring over pine cross sections images.

"""
from pathlib import Path
import time

from lib.io import load_image
from lib.preprocessing import preprocessing
from lib.canny_devernay_edge_detector import canny_deverney_edge_detector
from lib.filter_edges import filter_edges
from lib.sampling import sampling_edges
from lib.connect_chains import connect_chains
from lib.postprocessing import postprocessing
from lib.utils import chain_2_labelme_json, save_config, saving_results

def TreeRingDetection(im_in, cy, cx, height, width, sigma, low, high, edges_th, nr, mc, debug, image_input_path, output_dir):
    """
    Method for delineating tree ring over pine cross sections images.
    @param im_in: segmented input image
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param height: new image height_output size
    @param width: new image width_output size
    @param sigma: edge detection gaussian filtering
    @param low: gradient threshold low
    @param high: gradient threshold high
    @param edges_th: Edge filtering parameter. Collinearity threshold
    @param nr: rays number
    @param mc: min chain lenght
    @param debug: deb
    @param image_input_path: path to input image. Used to write labelme json.
    @param output_dir: Output directory. Debug results are saved here.
    @return:
     - im_pre: preprocessing image results
     - ch_e: Intermediate results. Devernay curves
     - ch_f: Intermediate results. Filtered Devernay curves
     - ch_s: Intermediate results. Sampled devernay curves as Chain objects
     - ch_c: Intermediate results. Chain lists after connect stage.
     - ch_p: Intermediate results. Chain lists after posprocessing stage.
     - rings: Final results. Json file with rings coordinates.
    """
    to = time.time()

    im_pre, cy, cx = preprocessing(im_in, height, width, cy, cx)
    ch_e, gx, gy = canny_deverney_edge_detector(im_pre, sigma, low, high)
    ch_f = filter_edges(ch_e, cy, cx, gx, gy, edges_th, im_pre)
    ch_s, nodes_s = sampling_edges(ch_f, cy, cx, nr, mc, im_pre, debug=debug)
    ch_c,  nodes_c = connect_chains(ch_s, nodes_s, cy, cx, nr, im_pre, debug, output_dir)
    ch_p = postprocessing(ch_c, nodes_c, cy, cx, output_dir, im_pre, debug)

    tf = time.time()
    rings = chain_2_labelme_json(ch_p, image_input_path, height, width, im_in, tf - to, cy, cx)
    return im_in, im_pre, ch_e, ch_f, ch_s, ch_c, ch_p, rings



def main(args):
    save_config(args, args.root, args.output_dir)
    im_in = load_image(args.input)
    Path(args.output_dir).mkdir(exist_ok=True)

    res = TreeRingDetection(im_in, args.cy, args.cx, args.hsize, args.wsize, args.sigma, args.th_low, args.th_high,
                      args.edge_th, args.nr, args.min_chain_length, args.debug, args.input, args.output_dir)

    saving_results(res, args.output_dir, args.save_imgs)

    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_imgs", type=str, required=False)
    parser.add_argument("--sigma", type=float, required=True,default=3)
    parser.add_argument("--nr", type=int, required=False,default=360)
    parser.add_argument("--hsize", type=int, required=False, default=None)
    parser.add_argument("--wsize", type=int, required=False, default=None)
    parser.add_argument("--edge_th", type=int, required=False, default=30)
    parser.add_argument("--th_high", type=int, required=False, default=20)
    parser.add_argument("--th_low", type=int, required=False, default=5)
    parser.add_argument("--min_chain_length", type=int, required=False, default=2)
    parser.add_argument("--debug", type=int, required=False)

    args = parser.parse_args()
    main(args)





