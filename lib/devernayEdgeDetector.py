#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:18:32 2022

@author: henry
"""

import os 
import subprocess
import copy

import pandas as pd
from skimage.util import view_as_windows
import numpy as np
from PIL import Image as im
import cv2 
import matplotlib.pyplot as plt 
from scipy import ndimage
from scipy.ndimage.filters import convolve
from skimage.exposure import rescale_intensity, equalize_adapthist

from lib.io import get_path,Nr
import lib.edges_filter as edges
import lib.chain_v4 as ch
import lib.edges_filter as edges 
from lib import utils as utils
from lib.io import pliegoGrados

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
    def __init__(self, nombre,img,centro,save_path, sigma=1, kernel_size=21, weak_pixel=75, strong_pixel=255, lowthreshold=5, highthreshold=15, debug=False):
        self.nombre = nombre
        self.s = 1.5
        self.l = lowthreshold
        self.h = highthreshold
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
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.centro = centro
        self.save_path = save_path
        
    def __convert_image_to_pgm(self,img):
        #self.display_image_matrix(img,self.centro, f"{self.save_path}/original.png", "original")
        #data = im.fromarray(img)
        # Convert to grey
        gray = img*255 #cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.uint8(gray)
        # Write to disk
        print(self.image_path)
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
        mod = np.zeros_like(self.img).astype(float)
        theta = np.zeros_like(self.img).astype(float)
        # Gx[1:-1,1:-1] = np.loadtxt(self.gx_path).T
        # Gy[1:-1,1:-1] = np.loadtxt(self.gy_path).T
        # mod[1:-1,1:-1] = np.loadtxt(self.mod_path).T

        Gx[1:-1,1:-1] = pd.read_csv(self.gx_path,delimiter=" ",header=None).values.T
        Gy[1:-1,1:-1] = pd.read_csv(self.gy_path,delimiter=" ",header=None).values.T
        #mod[1:-1,1:-1] = pd.read_csv(self.mod_path,delimiter=" ",header=None).values.T
        #el angulo se calcula asi.
        #theta = np.arctan2(Gy, Gx)
        return mod,theta,Gx,Gy

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
        #curve_bin = np.loadtxt(self.outputtxt)
        curve_bin = pd.read_csv(self.outputtxt,delimiter=" ",header=None).values
        counter_curves = 0
        curve_border_index = np.where(curve_bin == np.array([-1, -1]))[0]
        X = curve_bin.copy()
        X[curve_border_index] = 0
        #B = np.split(curve_bin, curve_border_index)
        #X = np.delete(curve_bin, curve_border_index, axis = 0)
        gradient = np.vstack(( self.Gx[X[:,1].astype(int), X[:,0].astype(int)], self.Gy[X[:,1].astype(int), X[:,0].astype(int)])).T

        Xb = np.array([ [1,0], [0,1]]).dot(X.T) + (np.array([-1,-1]) * np.array(self.centro[::-1],dtype=float)).reshape((-1, 1))

        Xb_normed = self.normalized_matrix(Xb.T)
        gradient_normed = self.normalized_matrix(gradient)

        theta = np.arccos(np.clip((gradient_normed * Xb_normed).sum(axis=1),-1.0,1.0)) * 180 / np.pi
        threshold = 90 - pliegoGrados
        X_edges_filtered = curve_bin.copy()
        X_edges_filtered[theta > threshold] = -1

        ###############################################
        #### formar vector de direcciones de rayos
        ###############################################
        # Vrayos = np.array([[np.sin(angulo*np.pi/180),np.cos(angulo*np.pi/180)] for angulo in np.arange(0,360,360/Nr)])
        # plt.figure()
        # plt.scatter(Vrayos[:,0],Vrayos[:,1])
        # plt.show()
        #
        # cos_dot = Vrayos.dot(Xb_normed.T)
        # idx_sampled = [[idx for idx in range(Xb_normed.shape[0]) if np.where(cos_dot[:,idx]==1)[0].shape[0]>0]
        # ###################################################
        # self.curves_list = []
        #
        # puntos_curva = []
        # for idx in range(X_edges_filtered.shape[0]):
        #     j, i = X_edges_filtered[idx][0], X_edges_filtered[idx][1]
        #     if j < 0 and i < 0 and len(puntos_curva)>2:
        #         curve = Curva(puntos_curva,id=counter_curves)
        #         self.curves_list.append(curve)
        #         puntos_curva = []
        #         counter_curves += 1
        #         continue
        #
        #     if not (j < 0 and i < 0):
        #         puntos_curva.append((j,i))
            #curve.add_pixel(i, j)

        #self.curves_list = [curve for curve in self.curves_list if curve.size > 0]
        #self._graficar_curvas_devernay("curvas_devernay_end_colors", labels=False, colors_type=False)
        #X sampled

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

        #self._graficar_curvas_devernay("curvas_devernay", labels=False, colors_type=False)
    
    @staticmethod
    def display_image_matrix(img,centro,save_path,title):
        # plt.figure(figsize=(30,30))
        # plt.imshow(img,cmap='gray')
        # plt.scatter(centro[1],centro[0])
        # plt.title(title)
        # plt.axis('off')
        # plt.savefig(save_path)
        # #plt.show()
        # plt.close()
        M,N = img.shape
        img_aux = np.zeros((M,N,3))
        y,x = np.where(img>0)
        img_aux[y,x] = (255,0,0)
        cv2.imwrite(save_path,img_aux)
    
    

    def convertirAlistaPuntos(self,perfiles):
        matriz = ch.dict2matriz(perfiles,self.centro)
        img_perfiles = np.zeros_like(self.img_bin)
        listaPuntos = []
        for i,j,angulo,radio,fase in matriz:
            img_perfiles[np.uint16(i),np.uint16(j)] = 1

        return img_perfiles
    
    
    def filter_by_phase(self):
        img_bin = np.where(self.img_bin>BACKGROUND_VALUE,1,0)
        #self.display_image_matrix(img_bin,self.centro,f"{self.save_path}/bordes_sin_procesar.png","binaria")
        binaria_filter_dark_light = self.filter_by_gradient_phase(img_bin)
        #self.display_image_matrix(binaria_filter_dark_light,self.centro,f"{self.save_path}/binaria_sin_claro_oscuro.png","binaria_sin_claro_oscuro")
        bin_sin_pliegos = self.filter_by_gradient_phase(binaria_filter_dark_light,threshold_high=90+pliegoGrados,threshold_low=90-pliegoGrados)
        #self.display_image_matrix(bin_sin_pliegos,self.centro,f"{self.save_path}/bordes_sin_pliegos.png","binaria_sin_pliegos")
        self.bin_sin_pliegos = bin_sin_pliegos

        
    
    @staticmethod
    def plot_image_gradient(I,dy,dx,title,step=10):
        #dy = Gx
        #dx = Gy
        w,h = I.shape
        #x, y = np.mgrid[0:h:1000j, 0:w:1000j]
        x = np.arange(0,w)
        y = np.arange(0,h)
        x, y = np.meshgrid(x, y)
        skip = (slice(None, None, step), slice(None, None, step))
        
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(I,#.transpose(Image.FLIP_TOP_BOTTOM), 
                       extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(im)
        ax.quiver(x[skip], y[skip], dx[skip], dy[skip])
        
        ax.set(aspect=1, title=title)
        #ax.set_aspect('equal')
        plt.show()
        plt.close()

    @staticmethod
    def calculate_vector_refered_to_center(i, j, i_c, j_c):
        v1_i = float(i-i_c)
        v1_j = float(j-j_c)
        v1 = np.array([v1_j, v1_i]).astype(float)

        return v1

    def borde_valido(self,i,j,threshold_low=100,threshold_high=181):
        i_int, j_int = int(i), int(j)
        i_c = self.centro[0]
        j_c = self.centro[1]
        if self.Gx[i_int, j_int] == 0 and self.Gy[i_int, j_int] == 0:
            return True
        v1 = self.calculate_vector_refered_to_center(i_int, j_int, i_c, j_c)
        v2 = np.array([self.Gx[i_int, j_int], self.Gy[i_int, j_int]]).astype(float)
        angle_between = edges.angle_between(v1, v2) * 180 / np.pi

        if threshold_low <= angle_between < threshold_high:
            return False
        else:
            return True
    def filter_by_gradient_phase(self,binaria,threshold_low=100,threshold_high=181,debug=False):
        Gx_copy = self.Gx.copy().astype(float)
        Gy_copy = self.Gy.copy().astype(float)
        binaria_copy = binaria.copy()
        centro = self.centro
        for curve_idx,curve in enumerate(self.curves_list):
            i_c=centro[0]
            j_c=centro[1]
            for pix_idx,pix in enumerate(curve.pixels_list):
                    i,j = int(pix.y),int(pix.x)
                    if Gx_copy[i,j] == 0 and Gy_copy[i,j]== 0:
                        continue
                    v1 = self.calculate_vector_refered_to_center(i,j,i_c,j_c)
                    v2 = np.array([Gx_copy[i,j],Gy_copy[i,j]]).astype(float)
                    angle_between = edges.angle_between(v1,v2)*180/np.pi

                    if threshold_low<= angle_between<threshold_high:
                            #Gx_copy[i,j] = 0
                            #Gy_copy[i,j] = 0
                            binaria_copy[i,j] = 0


        if debug:
            Gy_copy[binaria_copy==0] = 0
            Gx_copy[binaria_copy == 0] = 0
            self.plot_image_gradient(self.img, self.Gy, self.Gx,"Gradient")
            self.plot_image_gradient(self.img, Gy_copy, Gx_copy,"Gradient filtered")
            plt.figure()
            plt.imshow(binaria_copy)
            plt.title('binaria filtrada oscuro-claro')
            plt.show()



        return binaria_copy

    def _execute_command(self):
        command = f"{str(self.root_path)}/devernay  {self.image_path} -s {self.s} -l {self.l} -h {self.h} -t {self.outputtxt} -p {str(self.root_path)}/output.pdf -g {str(self.root_path)}/output.svg -n {self.nombre}"
        print(command)
        os.system(command)

    def angulo_pixel(self,i,j,centro):
        vector = np.array([float(i)-float(centro[0]), float(j)-float(centro[1])])
        radAngle = np.arctan2(vector[1], vector[0])* 180 / np.pi
        radAngle = radAngle if radAngle > 0 else radAngle + 360
        gradAngle = np.ceil(radAngle ) % 360
        return gradAngle

    def _converting_to_dot_chains_objects(self):
        y,x = np.where(self.matriz_puntos>0)
        matriz_angulos_computados = np.zeros_like(self.angulos_matrix)+BACKGROUND_VALUE
        listaPuntos = []
        from tqdm import tqdm
        for i,j in tqdm(zip(y,x)):
            params = {
                "x": i,
                "y": j,
                "angulo": np.ceil(self.angulos_matrix[i,j]),#ch.getAngleFromCoordinates(i, j, self.centro),
                "radio": ch.getRadialFromCoordinates(i, j, self.centro[::-1]),
                "gradFase": self.thetaMat[i, j],
                "cadenaId": self.img_bin[i,j],
            }
            matriz_angulos_computados[i,j] = self.angulo_pixel(i,j,self.centro)#ch.getAngleFromCoordinates(i, j, self.centro)
            punto = ch.Punto(**params)
            #angulo_puntos_misma_cadenas = [dot.angulo for dot in listaPuntos if dot.cadenaId==self.img_bin[i,j]]
            # if punto.angulo not in angulo_puntos_misma_cadenas:
            #     listaPuntos.append(punto)
            # else:
            #     self.matriz_puntos[i,j]=0
            if punto in listaPuntos:
                continue
            listaPuntos.append(punto)

        ii,jj = np.where(matriz_angulos_computados>BACKGROUND_VALUE)
        assert np.sum(matriz_angulos_computados[ii,jj] - np.ceil(self.angulos_matrix[ii,jj])) == 0

        listaCadenas = []
        etiquetas = list(np.unique(self.img_bin))
        if BACKGROUND_VALUE in etiquetas:
            etiquetas.remove(BACKGROUND_VALUE)

        M,N = self.img_bin.shape
        for label in tqdm(etiquetas):
            puntos_pertenecientes_a_cadena = [dot for dot in listaPuntos if dot.cadenaId==label]
            if len(puntos_pertenecientes_a_cadena)==0:
                continue
            cadena = ch.Cadena(label,self.centro,M,N)
            cadena.add_lista_puntos(puntos_pertenecientes_a_cadena)

            if cadena.size>0:
                listaCadenas.append(cadena)

        #####################################################################
        #limpiar cadenas de largo 1
        for cadena in tqdm(listaCadenas):
            if cadena.size < 2:
                for dot in cadena.lista:
                    listaPuntos.remove(dot)
                    self.matriz_puntos[dot.x,dot.y]=0
        listaCadenas = [cad for cad in listaCadenas if cad.size > 1]
        #######################################################################

        ch.visualizarCadenasSobreDisco(
            listaCadenas, self.img, "cadenas_completas", labels=True,save=self.save_path
        )

        return listaCadenas, listaPuntos

    def _delete_files(self):
        files = [ self.outputtxt, self.image_path, self.gx_path, self.gy_path, self.mod_path, self.non_max_path,
                  f"{str(self.root_path)}/output.pdf", f"{str(self.root_path)}/output.svg"]
        for file in files:
            os.system(f"rm {file}")

    def put_text(self,text,image,org):
        # font
        font = cv2.FONT_HERSHEY_DUPLEX
        # fontScale
        fontScale = 1/3

        # Blue color in BGR
        color = (255,255, 255)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(image, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        return image
    def _graficar_curvas_devernay(self,name,labels=True,colors_type=True):
        M,N = self.img_bin.shape
        img_curvas = np.zeros((M,N,3))
        img_curvas[:,:,0] = (self.img.copy()*255).astype(np.uint8)
        img_curvas[:, :, 1] = (self.img.copy() * 255).astype(np.uint8)
        img_curvas[:, :, 2] = (self.img.copy() * 255).astype(np.uint8)

        colors_length = 20
        colors = np.random.randint(low=0,high=256, size=(colors_length,3),dtype=np.uint8)
        color_idx  = 0
        for curve in self.curves_list:
            # print(str(curve.id))
            # if curve.id != 208:
            #     continue
            x = np.array([pix.x for pix in curve.pixels_list])
            y = np.array([pix.y for pix in curve.pixels_list])
            pts = np.vstack((x,y)).T.astype(int)
            isClosed = False
            thickness = 2
            if colors_type:
                b,g,r = (0,255,0)
            else:
                b,g,r = colors[color_idx]
            img_curvas = cv2.polylines(img_curvas, [pts],
                                isClosed, (int(b),int(g),int(r)), thickness)
            color_idx = (color_idx + 1) % colors_length

            #put text
            if labels:
                org = curve.pixels_list[0]
                img_curvas = self.put_text(str(curve.id),img_curvas,(int(org.x),int(org.y)))

        cv2.imwrite(f"./edge_detector.png",img_curvas)



    def logica_procesamiento_curva(self,curva,id,debug=False):
        #borde_de_interes = lambda i,j: self.borde_valido(i,j) and self.borde_valido(i,j,threshold_high=90+pliegoGrados,threshold_low=90-pliegoGrados)
        borde_de_interes = lambda i,j: self.borde_valido(i,j,threshold_low=90-pliegoGrados)
        lista_curvas = []
        nueva_curva = Curve(id)
        for idx,pix in enumerate(curva.pixels_list):
            pixel_pertenece_a_borde = borde_de_interes(pix.y,pix.x)
            if pixel_pertenece_a_borde:
                nueva_curva.add_pixel(pix.y,pix.x)
            else:
                if nueva_curva.size > 0:
                    lista_curvas.append(nueva_curva)
                    nueva_curva = Curve(nueva_curva.id+1)

        if nueva_curva.size > 0:
            lista_curvas.append(nueva_curva)

        if debug:
            M, N = self.bin_sin_pliegos.shape
            img_curvas = np.zeros((M, N, 3)) + (255,255,255)

            x = np.array([pix.x for pix in curva.pixels_list])
            y = np.array([pix.y for pix in curva.pixels_list])
            pts = np.vstack((x, y)).T.astype(int)
            isClosed = False
            thickness = 5
            b, g, r = (0,0,255)
            img_curvas = cv2.polylines(img_curvas, [pts],
                                       isClosed, (int(b), int(g), int(r)), thickness)
            plt.figure();
            plt.imshow(img_curvas);
            plt.show()

        return lista_curvas

    def procesar_curvas(self):
            if self.debug:
                self._graficar_curvas_devernay("curvas_devernay_init",labels=False,colors_type=False)
            lista_curvas_procesadas = []
            for curve in self.curves_list:
                lista_curvitas = self.logica_procesamiento_curva(curve,len(lista_curvas_procesadas))
                lista_curvas_procesadas+=lista_curvitas

            self.curves_list = lista_curvas_procesadas
            if self.debug:
                self._graficar_curvas_devernay("curvas_devernay_end_labels",labels=True,colors_type=False)
                self._graficar_curvas_devernay("curvas_devernay_end", labels=False)
                self._graficar_curvas_devernay("curvas_devernay_end_colors", labels=False,colors_type=False)










    def buscar_radio_de_mayor_longitud(self,tramos_dic):
        rango_angulos = np.arange(0,Nr)
        #1.0 busco radio de mayor longitud
        maximo = 0
        direccion_mas_larga = 0
        for direccion in rango_angulos:
            maximo_actual = np.cumsum(tramos_dic[direccion])[-1]
            if maximo_actual>maximo:
                maximo = maximo_actual
        return maximo
    def sobremuestrear_tramos(self,tramos_dic,maximo,diviciones_radiales = Nr/2,debug=None):
        rango_angulos = np.arange(0, Nr)
        step = maximo / diviciones_radiales
        v1 = np.arange(0,maximo,step)
        matriz_tramos_sobremuestreada = np.zeros((len(v1),Nr))
        matriz_tramos_sobremuestreada[:,:] = np.nan
        # loop
        for direccion in rango_angulos:
            longitud_tramos = tramos_dic[direccion]
            suma_acumulada = np.cumsum(longitud_tramos)
            longitud_tramos_sobremuestreada = np.zeros_like(v1)
            for idx_tramo in range(len(longitud_tramos)):
                longitud = suma_acumulada[idx_tramo]
                if idx_tramo > 0:
                    longitud_anterior = suma_acumulada[idx_tramo-1]
                    desde = np.where(v1 <= longitud_anterior)[0][-1]
                else:
                    desde = 0
                elementos_mas_lejanos = np.where(v1 > longitud)[0]
                hasta = elementos_mas_lejanos[0] if len(elementos_mas_lejanos)>0 else -1
                longitud_tramos_sobremuestreada[desde:hasta] = longitud_tramos[idx_tramo]
            matriz_tramos_sobremuestreada[:, int(direccion)] = longitud_tramos_sobremuestreada
            print(f"direccion {direccion},minimo={np.min(longitud_tramos_sobremuestreada[np.where(longitud_tramos_sobremuestreada>0)[0]])}")
            if debug is not None:
                if direccion in debug:
                    plt.figure()
                    plt.title(f'Direccion={direccion}')
                    plt.plot(longitud_tramos_sobremuestreada)
                    # single line
                    xx = [np.where(v1<suma_acumulada[idx])[0][-1] for idx,longitud in enumerate(longitud_tramos)]
                    plt.vlines(x=xx, ymin=0, ymax=np.max(longitud_tramos),
                               colors='purple')
                    plt.show()

        return matriz_tramos_sobremuestreada,v1
    def sobremuestrear_superficie_tramos(self,tramos_dic):
        maximo = self.buscar_radio_de_mayor_longitud(tramos_dic)
        #2.0
        puntos_radiales = Nr
        matriz_tramos_sobremuestreados,v1 = self.sobremuestrear_tramos(tramos_dic,maximo,diviciones_radiales=puntos_radiales,debug=[0,90,180,270])
        plt.figure();plt.title('Distancia Tramo (radio//2)');plt.plot(matriz_tramos_sobremuestreados[int(puntos_radiales//2),:]);plt.show();plt.close()
        return matriz_tramos_sobremuestreados,v1

    def graficar_matriz_polares(self, th, r, z, nombre_archivo, proyection='polar'):
        fig = plt.figure()
        if proyection in 'polar':
            ax = fig.add_subplot(111, projection='polar')
        else:
            ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(th, r, z, cmap=plt.get_cmap('Spectral'))
        # plt.grid()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if proyection in 'polar':
            ax.set_theta_zero_location('S')
        fig.colorbar(pcm, ax=ax, orientation="vertical")
        plt.savefig(f"{self.save_path}/{nombre_archivo}")
        plt.close()

    def submuestrear(self,longitudes_direccion_radial_sobremuestreada):
        longitudes_direccion_radial = []
        idx_aux = 0
        ultimo_tramo = -1
        while idx_aux < len(longitudes_direccion_radial_sobremuestreada):
            tramo_actual = longitudes_direccion_radial_sobremuestreada[idx_aux]
            if tramo_actual == ultimo_tramo:
                idx_aux += 1
                continue
            longitudes_direccion_radial.append(tramo_actual)
            ultimo_tramo = tramo_actual
            idx_aux += 1

        while 0 in longitudes_direccion_radial:
            longitudes_direccion_radial.remove(0)

        return longitudes_direccion_radial

    def longitud_tramos_a_coordenadas_imagen(self, direccion, posicion_relativa_vector_muestreo,vector_muestreo):
        # coordenadas de los pixeles en la direccion radial
        coordenadas_direccion = self.get_radial_coordinates(direccion)
        # calcular modulo pixeles
        modulo = np.sqrt((coordenadas_direccion[:, 0] - self.centro[0]) ** 2 + (
                coordenadas_direccion[:, 1] - self.centro[1]) ** 2)
        # muestrear modulo segun vector muestreo
        # muestrear radio
        coor_step = [list(self.centro)]
        for step in vector_muestreo[1:]:
            elementos = np.where(modulo >= step)[0]
            if len(elementos) == 0:
                break
            coor_step.append(list(coordenadas_direccion[elementos[0]]))
        coor_step = np.array(coor_step)
        return coor_step[posicion_relativa_vector_muestreo]
        # obtener coordenadas referidos a la direccion radial sobre imagen
        # coordenadas_pixeles = []
        # for idx in posicion_relativa_vector_muestreo:
        #     longitud = np.round(vector_muestreo[idx])
        #     mayores = np.where(modulo >= longitud)[0]
        #     if len(mayores) == 0:
        #         continue
        #
        #     coord_pixel = mayores[0]
        #     coordenadas_pixeles.append(list(coordenadas_direccion[coord_pixel]))
        return np.array(coordenadas_pixeles)

    def interseccion_radio_vs_bordes(self,ii,jj):
        ii_f,jj_f=[],[]
        idx=0
        contador=0
        for i,j in zip(ii,jj):
            es_borde = np.where(self.bin_sin_pliegos[i, j]>0)[0]
            if es_borde.shape[0]>0:
                ii_f.append(i)
                jj_f.append(j)
                idx+=1
                continue

            if idx>=len(ii)-1:
                continue
            next_i,next_j = ii[idx+1],jj[idx+1]
            es_borde = np.where(self.bin_sin_pliegos[next_i, next_j] > 0)[0]
            if es_borde.shape[0]>0:
                idx+=1
                continue

            i_min = np.minimum(i,next_i)
            i_max = np.maximum(i,next_i)
            j_min = np.minimum(j,next_j)
            j_max = np.maximum(j,next_j)
            es_borde = np.where(self.bin_sin_pliegos[i_min:i_max+1,j_min:j_max+1]>0)[0]
            if len(es_borde)>0:
                ii_f.append(i)
                jj_f.append(j)
                idx += 1
                contador+=1
                continue
            idx+=1

        return np.array(ii_f),np.array(jj_f)
    def construir_superficie_tramos(self):
        tramos_dic = {}
        rango_angulos = np.arange(0,360,360/Nr)
        matriz_puntos_original = np.zeros_like(self.bin_sin_pliegos)
        self.matriz_rayos = np.zeros_like(self.bin_sin_pliegos)
        for direccion in rango_angulos:
            coordenadas_direccion = self.get_radial_coordinates(direccion)
            ii,jj = coordenadas_direccion[:,0],coordenadas_direccion[:,1]
            self.matriz_rayos[ii,jj] = 255
            ii,jj = self.interseccion_radio_vs_bordes(ii,jj)

            #distancia entre dos pixeles consecutivos mayor a 1
            ii_f,jj_f = [ii[0]],[jj[0]]
            idx_min = 0
            for i,j in zip(ii[1:],jj[1:]):
                dist = self.distancia_entre_pixeles(ii_f[-1],jj_f[-1],i,j)
                if dist>np.sqrt(2):
                    ii_f.append(i)
                    jj_f.append(j)
                idx_min+=1
            ii,jj = np.array(ii_f),np.array(jj_f)
            #ordenar distancia al centro
            distancias = [self.distancia_entre_pixeles(self.centro[0], self.centro[1], i, j) for i, j in zip(ii, jj)]
            sort_index = np.argsort(distancias)
            ii,jj=ii[sort_index],jj[sort_index]
            matriz_puntos_original[ii,jj] = 255

            #calcular longitudes tramos
            longitud_tramos = [self.distancia_entre_pixeles(self.centro[0],self.centro[1],ii[0],jj[0])]
            idx = 0
            for i,j in zip(ii[1:],jj[1:]):
                longitud_tramos.append(self.distancia_entre_pixeles(ii[idx],jj[idx],i,j))
                idx+=1

            tramos_dic[direccion] = longitud_tramos


        plt.figure(figsize=(15, 15));
        M,N = self.img.shape
        frame = np.zeros((M,N,3))
        yy, xx = np.where(self.bin_sin_pliegos > 0);
        frame[yy,xx,0] = 255
        plt.scatter(xx, yy, s=1);
        yy, xx = np.where(self.matriz_rayos > 0)
        frame[yy, xx, 1] = 255
        #plt.scatter(xx, yy, c='k', s=1)
        yy,xx = np.where(matriz_puntos_original>0)
        frame[yy, xx, :] = (0,0,255)
        cv2.imwrite(f"{self.save_path}/puntos_tramos_original.png",frame)
        plt.scatter(xx,yy,c='r',s=1)
        plt.gca().invert_yaxis();

        plt.savefig(f"{self.save_path}/puntos_tramos_original_sc.png")
        plt.close()
        #plt.show()

        return tramos_dic,matriz_puntos_original
    def reconstruccion_puntos_basado_en_tramos(self,matriz_tramos_filtrada_tangencial,vector_muestreo):
        rango_angulos = np.arange(0,360,360/Nr)
        matriz_puntos_recontruidos = np.zeros_like(self.angulos_matrix)
        for direccion in rango_angulos.astype(int):
            longitudes_direccion_radial_sobremuestreada_filtrada = list(matriz_tramos_filtrada_tangencial[:, direccion])

            ###obtener tramos ordenados
            f_x_plus_1 = np.array(longitudes_direccion_radial_sobremuestreada_filtrada[1:])
            f_x = np.array(longitudes_direccion_radial_sobremuestreada_filtrada[:-1])
            posicion_relativa_vector_muestreo = np.where(np.abs(f_x-f_x_plus_1)>0)[0] +1
            if direccion in []:
                plt.figure();
                plt.plot(vector_muestreo,longitudes_direccion_radial_sobremuestreada_filtrada);
                plt.vlines(vector_muestreo[posicion_relativa_vector_muestreo], ymin=0,
                           ymax=np.max(longitudes_direccion_radial_sobremuestreada_filtrada), color='r');plt.show()

            coordenadas_pixeles = self.longitud_tramos_a_coordenadas_imagen(direccion, posicion_relativa_vector_muestreo, vector_muestreo)
            # marcar pixeles reconstruidos en imagen
            matriz_puntos_recontruidos[coordenadas_pixeles[:, 0], coordenadas_pixeles[:, 1]] = 255

        return matriz_puntos_recontruidos

    def muestrear_curvas(self,step=1,distancia_radial_minima=5):
        cte_grados_a_radianes = np.pi/180
        matriz_puntos = np.zeros_like(self.img_bin)
        M,N = self.img_bin.shape
        xx, yy = np.meshgrid(np.arange(0, N),np.arange(0, M))
        yy_c,xx_c = yy - self.centro[0],xx - self.centro[1]
        self.angulos_matrix = np.arctan2(xx_c, yy_c) / cte_grados_a_radianes
        self.angulos_matrix = np.where(self.angulos_matrix>0, self.angulos_matrix,self.angulos_matrix + 360)
        self.angulos_matrix%=360
        rango_angulos = np.arange(0,360,step)
        M,N = matriz_puntos.shape
        matriz_tramos = np.zeros((Nr,Nr))
        tramos_dic = {}
        for angulo in rango_angulos:
            #pixeles dentro de rango angular
            ii, jj = np.where(np.ceil(self.angulos_matrix)==angulo)
            #ordenar pixeles desde el centro hacia fuera
            sort_index = np.argsort(jj-self.centro[1])
            ii,jj=ii[sort_index],jj[sort_index]
            #filtrar_para_quedarme con los bordes unicamente
            bordes_idx = self.bin_sin_pliegos[ii,jj]>0
            ii,jj = ii[bordes_idx],jj[bordes_idx]
            #filtrar para quedarme unicamente con un pixel por curva
            pixs_curve_id = self.img_bin[ii,jj]
            curvas_ids = list(np.unique(pixs_curve_id))
            if BACKGROUND_VALUE in curvas_ids:
                curvas_ids.remove(BACKGROUND_VALUE)
            ii_f,jj_f = [],[]
            for curva in curvas_ids:
                idx = np.where(pixs_curve_id == curva)[0]
                angulos_pix_curva = self.angulos_matrix[ii[idx], jj[idx]]
                cercano_angulo_medio = np.argsort(np.abs(angulos_pix_curva-(angulo+step*0.5)))[0]
                ii_f.append(ii[idx[cercano_angulo_medio]])
                jj_f.append(jj[idx[cercano_angulo_medio]])

            ii,jj = np.array(ii_f),np.array(jj_f)
            # distancia minima radial entre elementos muestreados de distancia_radial_minima
            lista_pixeles = []
            indice = 0
            for i,j in zip(ii,jj):
                pix2 = Pixel(x=j, y=i,id=curvas_ids[indice])
                indice+=1
                if len(lista_pixeles)>0:
                    pix1 = lista_pixeles[-1]
                    if distancia_euclidea(pix2,pix1) <= distancia_radial_minima:
                        #dejamos el pixel con curva de mayor tamaño
                        curve_1_size = len(self.img_bin[self.img_bin==pix1.id])
                        curve_2_size = len(self.img_bin[self.img_bin==pix2.id])
                        if curve_1_size<curve_2_size:
                            lista_pixeles.remove(pix1)
                            lista_pixeles.append(pix2)

                        continue

                lista_pixeles.append(pix2)
                matriz_puntos[i,j] = 1

            #verificacion que se cumple distancia minima
            distancias = []
            pix0 = lista_pixeles[0]
            for pix in lista_pixeles[1:]:
                distancias.append(distancia_euclidea(pix0,pix))
                pix0=pix

            distancia_minima = np.min(distancias)

            #tramo
            #
            # distancias = [self.distancia_entre_pixeles(self.centro[0], self.centro[1], i, j) for i, j in zip(ii, jj)]
            # sort_index = np.argsort(distancias)
            # ii,jj=ii[sort_index],jj[sort_index]
            # longitud_tramos = []
            # idx = 0
            #
            # for i,j in zip(ii,jj):
            #     if idx==0:
            #         lt = self.distancia_entre_pixeles(i,j,self.centro[0],self.centro[1])
            #         longitud_tramos.append(lt)
            #         idx += 1
            #         continue
            #     lt = np.sqrt((i-ii[idx-1])**2+ (j-jj[idx-1])**2)
            #     longitud_tramos.append(lt)
            #     idx+=1
            #
            # tramos_dic[angulo] = longitud_tramos
            # x = np.array(distancias)[sort_index]
            # sobre_muestreo = np.floor(Nr / len(x))
            # sobre_muestrear_longitud_tramos = np.repeat(longitud_tramos,sobre_muestreo)
            #
            # matriz_tramos[:sobre_muestrear_longitud_tramos.shape[0],int(angulo)] = sobre_muestrear_longitud_tramos


            print(f"radio {angulo} distancia_minima {int(distancia_minima)}")
            if distancia_minima<=distancia_radial_minima:
                raise


        #Algoritmo extraccion vectores tramos
        tramos_dic,matriz_puntos_original = self.construir_superficie_tramos()

        #Algoritmo sobremuestreo
        matriz_tramos_sobremuestreados,vector_muestreo = self.sobremuestrear_superficie_tramos(tramos_dic)

        ######

        self.graficar_puntos(matriz_puntos_original)
        rad = np.linspace(0, Nr,Nr)
        theta = np.linspace(0, 2 * np.pi, Nr )
        th, r = np.meshgrid(theta,rad)
        # z = matriz_tramos.copy()
        #self.graficar_matriz_polares(th,r,z,'tramos_polares_circulo.png',proyection='polar')
        #self.graficar_matriz_polares(th, r, z, 'tramos_polares_matriz.png', proyection='matriz')
        self.graficar_matriz_polares(th, r, matriz_tramos_sobremuestreados, 'tramos_polares_matriz_sobremuestreada.png', proyection='matriz')
        self.graficar_matriz_polares(th, r, matriz_tramos_sobremuestreados, 'tramos_polares_circulo_sobremuestreada.png',
                                     proyection='polar')


        #Algoritmo filtrado
        vecindad = 5
        matriz_tramos_filtrada_tangencial = self.filtro_mediana_tangencial(matriz_tramos_sobremuestreados, vecindad)
        #matriz_tramos_filtrada_tangencial = matriz_tramos_sobremuestreados
        self.graficar_matriz_polares(th, r, matriz_tramos_filtrada_tangencial, 'tramos_polares_circulo_sobremuestreada_filtrada.png',
                                     proyection='polar')
        self.graficar_matriz_polares(th, r, matriz_tramos_filtrada_tangencial, 'tramos_polares_matriz_sobremuestreada_filtrada.png',
                                     proyection='matriz')
        ####


        #Algoritmo recontruccion de puntos
        matriz_puntos_recontruidos = self.reconstruccion_puntos_basado_en_tramos(matriz_tramos_filtrada_tangencial,vector_muestreo)



        #obtener paso radial
        matriz_muestreo_rayos = np.zeros_like(self.matriz_rayos)
        for direccion in rango_angulos:
            # coordenadas de los pixeles en la direccion radial
            coordenadas_direccion = self.get_radial_coordinates(direccion)
            # calcular modulo pixeles
            modulo = np.sqrt((coordenadas_direccion[:, 0] - self.centro[0]) ** 2 + (
                    coordenadas_direccion[:, 1] - self.centro[1]) ** 2)
            #muestrear radio
            coor_step= []
            for step in vector_muestreo[1:]:
                elementos = np.where(modulo >= step)[0]
                if len(elementos) == 0:
                    break
                coor_step.append(list(coordenadas_direccion[elementos[0]]))
            coor_step = np.array(coor_step)
            matriz_muestreo_rayos[coor_step[:,0],coor_step[:,1]] = 255
        plt.figure(figsize=(15, 15));
        frame = np.zeros((M, N, 3))
        yy, xx = np.where(self.bin_sin_pliegos > 0);
        frame[yy, xx, 0] = 255
        plt.scatter(xx, yy, s=1);
        yy, xx = np.where(self.matriz_rayos > 0)
        frame[yy, xx, 1] = 255
        yy, xx = np.where(matriz_muestreo_rayos > 0)
        frame[yy, xx, :] = (255, 255, 255)
        yy,xx = np.where(matriz_puntos_recontruidos>0)
        frame[yy, xx, :] = (0, 0, 255)
        plt.scatter(xx,yy,c='r',s=1)

        plt.gca().invert_yaxis();
        plt.savefig(f"{self.save_path}/puntos_tramos_reconstruidos_sc.png")
        cv2.imwrite(f"{self.save_path}/puntos_tramos_reconstruidos.png", frame)
        plt.close()








        return matriz_puntos

    def distancia_entre_pixeles(self,i1, j1, i2, j2):
        return np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)

    def get_radial_coordinates(self,alpha):
        M,N = self.angulos_matrix.shape
        theta = alpha * np.pi / 180
        ii, jj = np.where(np.ceil(self.angulos_matrix) == alpha)
        unit_vect = np.array([np.cos(theta),np.sin(theta)])
        modulo = np.sqrt((ii-self.centro[0])**2+(jj-self.centro[1])**2).reshape(-1,1)
        radial_vectors = np.repeat(unit_vect.reshape(1,2),modulo.shape[0],axis=0) * modulo
        #me quedo con elementos unicos
        radial_coordinates = np.unique(radial_vectors.astype(int),axis=0) + self.centro

        #remover pixeles fuera de rango
        radial_coordinates_filtered = []
        ii,jj=[],[]
        for i,j in zip(radial_coordinates[:,0],radial_coordinates[:,1]):
            if 0<=i<M and 0<=j<N:
                ii.append(i)
                jj.append(j)

        #ordenar con respecto al centro
        ii,jj = np.array(ii),np.array(jj)
        distancias = [self.distancia_entre_pixeles(self.centro[0], self.centro[1], i, j) for i, j in zip(ii, jj)]
        sort_index = np.argsort(distancias)
        ii, jj = ii[sort_index], jj[sort_index]
        radial_coordinates_filtered = np.vstack((ii, jj)).T

        return radial_coordinates_filtered




    def filtro_mediana(self,img,kernel_size):
        padding = np.floor(kernel_size/2).astype(int)
        img_pad = np.pad(img,[(padding,padding),(padding,padding)],mode='wrap')
        #TODO: el borde superior y el inferior de l
        M,N = img.shape
        windows_img = view_as_windows(img_pad, (kernel_size, kernel_size))
        median_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                neighbourhood = windows_img[i,j]
                if i == 0:
                    median_img[i,j] = np.median(neighbourhood[1:,:])
                elif i == (img.shape[0]-1):
                    median_img[i, j] = np.median(neighbourhood[:-1, :])
                else:
                    median_img[i, j] = np.median(neighbourhood)
        return median_img

    def filtro_mediana_tangencial(self,img,kernel_size):
        padding = np.floor(kernel_size/2).astype(int)
        img_pad = np.pad(img,[(padding,padding),(padding,padding)],mode='wrap')
        M,N = img.shape
        median_img = np.zeros_like(img)
        for i in range(M):
            for j in range(N):
                    i_pad = i+padding
                    j_pad = j+padding
                    neighbourhood = img_pad[i_pad, j_pad - padding:j_pad + padding + 1]
                    median_img[i, j] = np.median(neighbourhood)
        return median_img




    def position(self,i,j,argmax):
        W_j = np.array([-1,0,1,-1,0,1,-1,0,1])
        W_i = np.array([-1,-1,-1,0,0,0,1,1,1])
        return i+W_i[argmax], j+W_j[argmax]

    def graficar_puntos(self,matriz_puntos):
        plt.figure(figsize=(30,30))
        plt.imshow(matriz_puntos,cmap='gray')
        #y,x = np.where(matriz_puntos>0)
        #plt.scatter(x,y,s=2,c='r')
        plt.scatter(self.centro[1],self.centro[0], s=2, c='b')
        plt.savefig(f"{self.save_path}/puntos_pixel.png")
        plt.close()

        import itertools
        import matplotlib.cm as cm
        plt.figure(figsize=(30, 30))
        plt.imshow(matriz_puntos, cmap='gray')
        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        for angulo in np.arange(0,360,360/Nr):
            y, x = np.where(np.ceil(self.angulos_matrix) == angulo)
            plt.scatter(x, y, s=1, color=next(colors))


        y,x = np.where(matriz_puntos>0)
        plt.scatter(x,y,s=1,c='k')
        plt.scatter(self.centro[1], self.centro[0], s=2, c='b')
        plt.savefig(f"{self.save_path}/puntos_pixel_rojo.png")
        plt.close()

    def detect(self):
        self.img_smoothed = cv2.GaussianBlur(self.img, (self.kernel_size, self.kernel_size), sigmaX=self.sigma,sigmaY=self.sigma)
        self.__convert_image_to_pgm(self.img_smoothed)
        self._execute_command()
        self.gradientMat, self.thetaMat,self.Gx,self.Gy = self.gradient()
        self.__load_curve_points_to_image()
        #self.filter_by_phase()
        #self.procesar_curvas()
        listaPuntos = []
        listaCadenas = []
        self._delete_files()

        return np.where(self.img_bin>0,255,0),None,self.thetaMat ,self.nonMaxImg,self.gradientMat, self.Gx, self.Gy, self.img_bin,listaCadenas,listaPuntos,self.curves_list

from pathlib import Path
from lib.io import get_path
if __name__=="__main__":
    image_file_name = '/data/maestria/datasets/FOTOS_DISCOS_1/segmentadas/F4A_CUT_up.tif'
    path = get_path('results') / 'deverernay'
    path.mkdir(exist_ok=True)
    #image_file_name = "/media/data/maestria/datasets/artificiales/segmentadas/example_10.tif"
    image = cv2.imread(image_file_name)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    rotated = image
    sigma = 1.4
    high=15
    low=5
    centro = [1204,1264]
    centro = [466,472]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img_seg = utils.segmentarImagen(gray, debug=False)
    img_eq = equalize_adapthist(np.uint8(gray), clip_limit=0.03)
    detector = devernayEdgeDetector(10,img_eq,centro[::-1],str(path), sigma=sigma,highthreshold=high,lowthreshold=low)
    img_bin,bin_sin_pliegos,thetaMat ,nonMaxImg,gradientMat,Gx,Gy,img_labels,listaCadenas,listaPuntos,lista_curvas = detector.detect()
    print(f"cantidad cadenas {len(listaCadenas)} cantidad puntos {len(listaPuntos)}")
        

    
    
    
    
    