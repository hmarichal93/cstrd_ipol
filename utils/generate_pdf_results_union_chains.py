#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:42:24 2022

@author: henry
"""

import glob
import argparse
from natsort import natsorted
from fpdf import FPDF
from tqdm import tqdm


def main(path):
        pdf = FPDF()
        pdf.set_font('Arial', 'B', 16)

        resultado_final = "output.png"
        figures = glob.glob(f"{path}/**/{resultado_final}", recursive=True)
        idx = 0
        for fig in tqdm(natsorted(figures)):
                idx += 1
                splitted = fig.split("/")
                disco = splitted[-2]
                disco = disco.replace("_","")

                fig_path = ""
                for sub_path in splitted[:-1]:
                    fig_path +=sub_path + "/"

                #fig2 = fig_path + "preprocessing_output.png"
                #fig3 = fig_path + "chains.png"
                fig4 = fig_path + "connect.png"
                fig5 = fig_path + "postprocessing.png"
                fig6 = fig_path + "output.png"
                x,y = 0,40
                height = 140

                for fig in [fig4, fig5, fig6]:
                        pdf.add_page()
                        pdf.cell(0, 0, disco)
                        pdf.image(fig,x,y, h=height)

                pdf.add_page()


        pdf.output(f"{path}/summary_ipol.pdf",'F')

if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--root_dir", type=str, required=True)

        args = parser.parse_args()

        path = args.root_dir

        main(path)

