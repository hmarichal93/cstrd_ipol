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

def TreeRingDetection(im_in, cy, cx, sigma, th_low, th_high, height, width, alpha, nr, mc, debug,
                      debug_image_input_path, debug_output_dir):
    """
    Method for delineating tree ring over pine cross sections images.
    @param im_in: segmented input image. Background must be white (255,255,255).
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param sigma: Canny edge detector gausssian kernel parameter
    @param th_low: Low threshold on the module of the gradient. Canny edge detector parameter.
    @param th_high: High threshold on the module of the gradient. Canny edge detector parameter.
    @param height: img_height of the image after the resize step
    @param width: width of the image after the resize step
    @param alpha: Edge filtering parameter. Collinearity threshold
    @param nr: rays number
    @param mc: min ch_i length
    @param debug: boolean, debug parameter
    @param debug_image_input_path: Debug parameter. Path to input image. Used to write labelme json.
    @param debug_output_dir: Debug parameter. Output directory. Debug results are saved here.
    @return:
     - l_rings: Final results. Json file with rings coordinates.
     - im_pre: Debug Output. Preprocessing image results
     - m_ch_e: Debug Output. Intermediate results. Devernay curves in matrix format
     - l_ch_f: Debug Output. Intermediate results. Filtered Devernay curves
     - l_ch_s: Debug Output. Intermediate results. Sampled devernay curves as Chain objects
     - l_ch_s: Debug Output. Intermediate results. Chain lists after connect stage.
     - l_ch_p: Debug Output. Intermediate results. Chain lists after posprocessing stage.
    """
    to = time.time()

    im_pre, cy, cx = preprocessing(im_in, height, width, cy, cx)
    m_ch_e, gx, gy = canny_deverney_edge_detector(im_pre, sigma, th_low, th_high)
    l_ch_f = filter_edges(m_ch_e, cy, cx, gx, gy, alpha, im_pre)
    l_ch_s, l_nodes_s = sampling_edges(l_ch_f, cy, cx, nr, mc, im_pre, debug=debug)
    l_ch_c,  l_nodes_c = connect_chains(l_ch_s, l_nodes_s, cy, cx, nr, debug, im_pre, debug_output_dir)
    l_ch_p = postprocessing(l_ch_c, l_nodes_c, cy, cx, debug, debug_output_dir, im_pre)
    debug_execution_time = time.time() - to
    l_rings = chain_2_labelme_json(l_ch_p, height, width, cx, cy, im_in, debug_image_input_path, debug_execution_time)
    return im_in, im_pre, m_ch_e, l_ch_f, l_ch_s, l_ch_c, l_ch_p, l_rings



def main(args):
    save_config(args, args.root, args.output_dir)
    im_in = load_image(args.input)
    Path(args.output_dir).mkdir(exist_ok=True)

    res = TreeRingDetection(im_in, args.cy, args.cx, args.sigma, args.th_low, args.th_high, args.hsize, args.wsize,
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
    parser.add_argument("--sigma", type=float, required=False,default=3)
    parser.add_argument("--nr", type=int, required=False,default=360)
    parser.add_argument("--hsize", type=int, required=False, default=0)
    parser.add_argument("--wsize", type=int, required=False, default=0)
    parser.add_argument("--edge_th", type=int, required=False, default=30)
    parser.add_argument("--th_high", type=int, required=False, default=20)
    parser.add_argument("--th_low", type=int, required=False, default=5)
    parser.add_argument("--min_chain_length", type=int, required=False, default=2)
    parser.add_argument("--debug", type=int, required=False)

    args = parser.parse_args()
    main(args)





