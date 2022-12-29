#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:18:32 2022

@author: henry
"""

import os
import pandas as pd
import numpy as np
import cv2
import time
import warnings
warnings.filterwarnings("ignore")

from lib.io import get_path
import lib.chain_v4 as ch

BACKGROUND_VALUE=-1

class Pixel:
    def __init__(self,x,y,id=-1):
        self.y = y
        self.x = x
        self.id = id

    def __eq__(self, other):
        return self.y == other.y and self.x == other.x

    def __repr__(self):
        return (f'({self.y},{self.x},{self.id})\n')

def distancia_euclidea(pix1,pix2):
    return np.ceil(np.sqrt((pix1.y - pix2.y)**2 + (pix1.x - pix2.x)**2))
class Curve:
    def __init__(self,id):
        self.id = id
        self.pixels_list = []#[ Pixel(x=elem[1],y=elem[0]) for elem in pixels_array]
        self.size = 0

    def add_pixel(self,y,x):
        self.pixels_list.append(Pixel(x=x,y=y,id=self.id))
        self.size+=1

    def get_size(self):
        return len(self.pixels_list)


class devernayEdgeDetector:
    def __init__(self,img,centro,save_path, config, debug=False, nombre='0'):
        self.nombre = nombre
        self.s = config['sigma']
        self.l = config['th_low']
        self.h = config['th_high']
        self.edge_th = config['edge_th']
        self.debug = debug
        self.img = img
        self.root_path = get_path('devernay')
        self.home = get_path('home')
        self.outputtxt = f"{str(self.home )}/output_{nombre}.txt"
        self.image_path = f"{str(self.home )}/test_{nombre}.pgm"
        self.gx_path = f"{str(self.home )}/gx_{nombre}.txt"
        self.gy_path = f"{str(self.home )}/gy_{nombre}.txt"
        self.mod_path = f"{str(self.home )}/mod_{nombre}.txt"
        self.non_max_path = f"{str(self.home)}/nonMax_{nombre}.txt"
        self.centro = centro
        self.save_path = save_path

        
    def __convert_image_to_pgm(self,img):
        #self.display_image_matrix(img,self.centro, f"{self.save_path}/original.png", "original")
        #data = im.fromarray(img)
        # Convert to grey
        gray = img*255 #cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.uint8(gray)
        # Write to disk
        cv2.imwrite(self.image_path, gray)
        #data.save(f"{str(self.root_path)}/test.pgm"

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    def gradient(self):
        Gx = np.zeros_like(self.img).astype(float)
        Gy = np.zeros_like(self.img).astype(float)


        Gx[1:-1,1:-1] = pd.read_csv(self.gx_path,delimiter=" ",header=None).values.T
        Gy[1:-1,1:-1] = pd.read_csv(self.gy_path,delimiter=" ",header=None).values.T

        return Gx,Gy

    @staticmethod
    def normalized_matrix( matrix):
        sqrt = np.sqrt((matrix**2).sum(axis=1))
        normalized_array = matrix / sqrt[:, np.newaxis]
        return normalized_array
    def __load_curve_points_to_image(self):
        M, N = self.img.shape[0], self.img.shape[1] 
        self.img_bin = np.zeros((M,N),dtype=np.int32)+BACKGROUND_VALUE
        self.img_position = self.img_bin.copy()

        self.nonMaxImg = np.zeros_like(self.img_bin)

        curve_bin = pd.read_csv(self.outputtxt,delimiter=" ",header=None).values
        counter_curves = 0
        curve_border_index = np.where(curve_bin == np.array([-1, -1]))[0]
        X = curve_bin.copy()
        X[curve_border_index] = 0

        gradient = np.vstack(( self.Gx[X[:,1].astype(int), X[:,0].astype(int)], self.Gy[X[:,1].astype(int), X[:,0].astype(int)])).T
        Xb = np.array([ [1,0], [0,1]]).dot(X.T) + (np.array([-1,-1]) * np.array(self.centro,dtype=float)).reshape((-1, 1))

        Xb_normed = self.normalized_matrix(Xb.T)
        gradient_normed = self.normalized_matrix(gradient)

        theta = np.arccos(np.clip((gradient_normed * Xb_normed).sum(axis=1),-1.0,1.0)) * 180 / np.pi
        threshold = 90 - self.edge_th
        X_edges_filtered = curve_bin.copy()
        X_edges_filtered[theta > threshold] = -1


        #pixel_relative_position_to_curve=0
        self.curves_list = []
        curve = Curve(counter_curves)
        self.curves_list.append(curve)
        for idx in range(X_edges_filtered.shape[0]):
            j,i = X_edges_filtered[idx][0],X_edges_filtered[idx][1]
            if j<0 and i<0:
                counter_curves+=1
                #pixel_relative_position_to_curve = 0
                curve = Curve(counter_curves)
                self.curves_list.append(curve)
                continue

            i_img = int(i)#np.round(i).astype(int)
            j_img= int(j)#np.round(j).astype(int)
            self.img_bin[i_img,j_img] = counter_curves
            curve.add_pixel(i,j)

        self.curves_list = [curve for curve in self.curves_list if curve.size>0]

        return 0

    def _execute_command(self):
        command = f"{str(self.root_path)}/devernay  {self.image_path} -s {self.s} -l {self.l} -h {self.h} -t {self.outputtxt} -p {str(self.root_path)}/output.pdf -g {str(self.root_path)}/output.svg -n {self.nombre}"
        os.system(command)

    def _delete_files(self):
        files = [ self.outputtxt, self.image_path, self.gx_path, self.gy_path, self.mod_path, self.non_max_path,
                  f"{str(self.root_path)}/output.pdf", f"{str(self.root_path)}/output.svg"]
        for file in files:
            os.system(f"rm {file}")



    def detect(self):
        self.__convert_image_to_pgm(self.img)
        self._execute_command()
        self.Gx,self.Gy = self.gradient()
        self.__load_curve_points_to_image()
        self._delete_files()

        return  self.Gx, self.Gy, self.img_bin,self.curves_list


def main(datos):
    M, N, img, centro, SAVE_PATH= datos['M'], datos['N'], datos['img'], datos['centro'], datos['save_path']
    image = datos['img_prep']
    to = time.time()
    detector = devernayEdgeDetector(image, centro=centro, save_path=SAVE_PATH, config= datos['config'])
    Gx, Gy, img_labels, lista_curvas = detector.detect()
    #cv2.imwrite(f"{SAVE_PATH}/edge_detector.png", np.where(img_labels > 0, 255, 0).astype(np.uint8))
    ch.visualizarCadenasSobreDisco(
        [], np.where(img_labels > 0, 255, 0).astype(np.uint8),f"{SAVE_PATH}/edge_detector.png", labels=False, gris=True, color=True
    )
    tf = time.time()

    datos['tiempo_bordes'] = tf - to
    print(f'Edge Detector: {tf - to:.1f} seconds')
    datos['Gy'] = Gy
    datos['Gx'] = Gx
    datos['lista_curvas'] = lista_curvas
    return 0