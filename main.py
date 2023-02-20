#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:54:53 2021

@author: henry
@brief:

"""
from pathlib import Path
import time

from lib.io import write_json, load_image, load_config
from lib.segmentation import segmentation
from lib.preprocessing import preprocessing, resize
from lib.canny_devernay_edge_detector import canny_deverney_edge_detector
from lib.filter_edges import filter_edges
from lib.sampling import sampling_edges
from lib.connect_chains import connect_chains
from lib.postprocessing import postprocessing
from lib.chain import visualize_chains_over_image, visualize_selected_ch_and_chains_over_image_
from lib.utils import chain_2_labelme_json
import lib.chain as ch
def TreeRingDetection(im_in, cy, cx, height, width, sigma, low, high, edges_th, nr,mc, debug, image_input, output_dir):
    to = time.time()

    im_seg = segmentation(im_in)
    im_pre, cy, cx = preprocessing(im_seg, height, width, cy, cx)
    ch_e, gx, gy = canny_deverney_edge_detector(im_pre, sigma, low, high)
    ch_f = filter_edges(ch_e, cy, cx, gx, gy, edges_th, im_pre)
    ch_s, nodes_s = sampling_edges(ch_f, cy, cx, nr, mc, im_pre, debug=debug)
    ch_c,  nodes_c = connect_chains(ch_s, nodes_s, cy, cx, nr, im_pre, debug, output_dir)
    ch_p = postprocessing(ch_c, nodes_c, cy, cx, output_dir, im_pre, debug)

    tf = time.time()
    rings = chain_2_labelme_json(ch_p, image_input, height, width, im_in, tf-to)
    return im_seg, im_pre, ch_e, ch_f, ch_s, ch_c, ch_p, rings



def save_config(args, root_path, output_dir):
    config = load_config()

    config['result_path'] = output_dir

    if args.nr:
        config['Nr'] = args.nr

    if args.hsize and args.wsize:
        if args.hsize>0 and args.wsize>0:
            config['resize'] = [args.hsize, args.wsize]

    if args.min_chain_length:
        config["min_chain_lenght"] = args.min_chain_length

    if args.edge_th:
        config["edge_th"] = args.edge_th

    if args.sigma:
        config['sigma'] = args.sigma

    if args.th_high:
        config['th_high'] = args.th_high

    if args.th_low:
        config['th_low'] = args.th_low

    if args.debug:
        config['debug'] = True

    write_json(config, Path(root_path) / 'config/general.json')

    return 0
def saving_results( res, output_dir):
    im_seg, im_pre, ch_e, ch_f, ch_s, ch_c, ch_p, rings = res
    M,N,_ = im_seg.shape
    M_n, N_n = im_pre.shape
    if M != M_n:
        im_seg,_,_ = resize(im_seg, (M_n, N_n))
    visualize_chains_over_image(img=im_seg, filename=f"{output_dir}/segmentation.png")
    visualize_chains_over_image(img=im_pre, filename=f"{output_dir}/preprocessing.png")
    visualize_chains_over_image(img=im_pre, filename=f"{output_dir}/edges.png", devernay=ch_e)
    visualize_chains_over_image(img=im_pre, filename=f"{output_dir}/filter.png", filter=ch_f)
    visualize_chains_over_image(chain_list=ch_s, img=im_seg, filename=f"{output_dir}/chains.png")
    visualize_chains_over_image(chain_list=ch_c, img=im_seg, filename=f"{output_dir}/connect.png")
    visualize_chains_over_image(chain_list=ch_p, img=im_seg, filename=f"{output_dir}/postprocessing.png")
    visualize_chains_over_image(chain_list=[chain for chain in ch_p if chain.is_full() and chain.type not in
        [ ch.TypeChains.center, ch.TypeChains.border] ], img=im_seg, filename=f"{output_dir}/output.png")
    write_json(rings, f"{output_dir}/labelme.json")

    return

def main(args):
    save_config(args, args.root, args.output_dir)
    im_in = load_image(args.input)
    Path(args.output_dir).mkdir(exist_ok=True)

    res = TreeRingDetection(im_in, args.cy, args.cx, args.hsize, args.wsize, args.sigma, args.th_low, args.th_high,
                      args.edge_th, args.nr, args.min_chain_length, args.debug, args.input, args.output_dir)

    saving_results(res, args.output_dir)

    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

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





