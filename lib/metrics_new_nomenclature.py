#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:12:13 2022

@author: henry
"""
import pandas as pd
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, LineString
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point
import argparse
import warnings
warnings.filterwarnings("ignore")

from lib.io import get_path, load_json
from lib.sampling import build_rays, draw_ray_curve_and_intersections
import lib.chain as ch
from lib.utils import write_json
#import lib.metrics_sampling as sampling
INFINITO=np.nan

def load_image( image_name, disk_name=None):
    img_name = image_name if disk_name is None else disk_name
    data_path = get_path("data")
    dataset = pd.read_csv(f"/data/maestria/database_new_nomenclature/dataset_ipol.csv")
    #nroImagen = [idx for idx,image_file in enumerate(dataset['Imagen'].values) if (image_file[:-4] in img_name) and (img_name in image_file[:-4])][0]
    disco = dataset[dataset.Imagen == image_name]
    #disco = dataset.iloc[int(nroImagen)]
    path = f"/data/maestria/database_new_nomenclature/images"
    imagenNombre = disco["Imagen"]
    centro = tuple([int(disco["cy"]), int(disco["cx"])])
    return cv.imread(str(path + f"/{image_name}.png"),cv.COLOR_RGB2GRAY),centro

class Polygon_node(Polygon):
    def __init__(self, node_list):
        self.angles = [node.angle for node in node_list]
        self.node_list = node_list
        super().__init__([[node.y, node.x] for node in node_list])

FP_ID = -1
class AreaInfluencia:
    def __init__(self, gt, dt,output_dir, Nr, disk_name=None):
        self.gt_path = Path(gt)
        self.output_dir = output_dir
        self.dt_path = dt
        image_name = self.gt_path.parts[-1][:-5]
        image_name = self.gt_path.name.split('.')[0].split('_')[0]
        self.image_name = image_name
        self.img, self.centro = load_image(image_name, disk_name)
        self.angulos_matrix = self._build_angle_matrix()
        M,N,_ = self.img.shape
        self.rayos_img = build_radial_directions_matrix(np.zeros((M,N)),self.centro, Nr).astype(int)
        self.rays_list = build_rays(Nr, N, M, self.centro)
        self.gt_poly = self.load_ring_stimation(self.gt_path)
        gt_poly_samples = []
        cy, cx = self.centro
        for poly in self.gt_poly:
            gt_poly_samples.append(self._sampling_poly(poly, cy, cx, self.rays_list))
        self.gt_poly = gt_poly_samples

        self.dt_poly = self.load_ring_stimation(self.dt_path)
        dt_poly_samples = []
        for idx,poly in enumerate(self.dt_poly):
            dt_poly_samples.append(self._sampling_poly(poly, cy, cx, self.rays_list) )
        self.dt_poly = dt_poly_samples
        self.Nr = Nr

    @staticmethod
    def load_ring_stimation(path):
        try:
            json_content = load_json(path)
            anillos = []
            for ring in json_content['shapes']:
                anillos.append( Polygon(ring['points']))

        except FileNotFoundError:
            anillos = []

        return anillos

    def _build_angle_matrix(self):
        M, N,_ = self.img.shape
        xx, yy = np.meshgrid(np.arange(0, N), np.arange(0, M))
        yy_c, xx_c = yy - self.centro[1], xx - self.centro[0]
        cte_grados_a_radianes = np.pi / 180
        angulos_matrix = np.arctan2(xx_c, yy_c) / cte_grados_a_radianes
        angulos_matrix = np.where(angulos_matrix > 0, angulos_matrix,angulos_matrix + 360)
        angulos_matrix %= 360

        return angulos_matrix

    def _convert_poly_dict_to_poly_list(self,poly_d):
        poly_list = []
        for key in poly_d.keys():
            poly_list.append(poly_d[key])

        return poly_list

    def _add_poly_to_img(self,img,poly,color,thickness=1):
        isClosed = True
        x,y = poly.exterior.coords.xy
        pts = np.vstack((x, y)).T.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        return cv.polylines(img, [pts], isClosed, color, thickness)

    # def _sampling_poly(self,poly,step):
    #     aux_img = np.zeros_like(self.angulos_matrix)
    #     aux_img = self._add_poly_to_img(aux_img,poly,color=[255],thickness=2).astype(int)
    #     inter_img = np.logical_and(aux_img, self.rayos_img)
    #     y,x = np.where(inter_img>0)
    #
    #     #Procesar puntos para que por rayo solo quede un unico punto.
    #     id_rayo_list = list(self.rayos_img[y,x])
    #     list_filtered = []
    #     list_angulos = []
    #     for idx,rayo_id in enumerate(id_rayo_list):
    #         if rayo_id not in list_angulos:
    #             list_filtered.append(idx)
    #             list_angulos.append(rayo_id)
    #
    #     list_filtered.sort(key= lambda x: id_rayo_list[x])
    #     y_s,x_s = y[list_filtered],x[list_filtered]
    #
    #     sampled_poly = Polygon(np.vstack((x_s, y_s)).T.astype(np.int32))
    #     assert np.unique(id_rayo_list).shape[0] == len(y_s)
    #     return sampled_poly

    @staticmethod
    def _sampling_poly( poly, cy, cx, rays_list):
        #MeanDisk.draw_ray_curve_and_intersections( [radii for radii in self.rays_list if radii.direction ==93], [poly], [], [], self.img, './debug.png')
        intersection_list = MeanDisk.compute_intersections( poly, rays_list, cy, cx)
        intersection_list.sort(key = lambda x: x.angle)
        return Polygon_node(intersection_list)

    def get_radial_coordinates(self,alpha):
        theta = alpha * np.pi / 180
        ii, jj = np.where(np.ceil(self.angulos_matrix) == alpha)
        unit_vect = np.array([np.cos(theta),np.sin(theta)])
        modulo = np.sqrt((ii-self.centro[0])**2+(jj-self.centro[1])**2).reshape(-1,1)
        radial_vectors = np.repeat(unit_vect.reshape(1,2),modulo.shape[0],axis=0) * modulo
        #me quedo con elementos unicos
        radial_coordinates = np.unique(radial_vectors.astype(int),axis=0) + self.centro
        return radial_coordinates


    def _plot_gt_and_dt_polys(self,img,gt,dt,n=1,title=None):
        img_aux = img.copy()
        #gt
        for poly in gt:
            img_aux = self._add_poly_to_img(img_aux,  poly,color=(0,255,0),thickness=n)

        #dt
        for poly in dt:
            img_aux = self._add_poly_to_img(img_aux, poly, color=(255, 0, 0),thickness=n)

        plt.figure(figsize=(10,10));plt.imshow(img_aux);plt.axis('off');
        if title is not None:
            plt.title(title)
        plt.savefig(f'{self.output_dir}/dt_and_gt.png')
        #plt.show();
        plt.close()

    def extraer_coordenadas_poligonos(self,poly):
        x, y = poly.exterior.coords.xy
        pts = np.vstack((x, y)).T.astype(np.int32)
        return pts

    def get_punto_con_angulo_alpha_img_binaria(self,img,alpha):
        y,x = np.where(img>0)
        mask = np.where(np.round(self.angulos_matrix[y,x])==alpha)[0]
        pts = np.vstack((y[mask],x[mask])).T
        return pts

    def calcular_error_respecto_a_gt(self,infinito=INFINITO):
        self.mapa_color = np.zeros_like(self.angulos_matrix)-infinito
        for idx,dt in enumerate(self.dt_poly):
            pts_dt = self.extraer_coordenadas_poligonos(dt)
            if len(self.asignacion_entre_dt_y_gt)-1 < idx:
                continue

            gt_idx = self.asignacion_entre_dt_y_gt[idx]
            if gt_idx == FP_ID:
                continue
            gt = self.gt_poly[gt_idx]
            pts_gt = self.extraer_coordenadas_poligonos(self.gt_poly[gt_idx])

            for alpha in np.arange(0, 360, 360 / self.Nr):
                pts1_angulo = self.ptos_con_angulo_alpha(alpha, pts_dt)
                # if len(pts1_angulo) == 0:
                #     continue
                pts1_media = pts1_angulo[0]
                pts2_angulo = self.ptos_con_angulo_alpha(alpha, pts_gt)
                # if len(pts2_angulo) == 0:
                #     continue
                pts2_media = pts2_angulo[0]
                # 2.0 busco punto intermedio (media)
                radio_dt = self.calcular_radio(pts1_media)
                radio_gt = self.calcular_radio(pts2_media)
                diferencia = radio_dt-radio_gt
                self.mapa_color[pts2_media[1],pts2_media[0]] = diferencia
        self.graficar_mapa_color()
        return 0

    def calcular_rmse_entre_dt_y_gt(self,dt_poly,gt_poly):
        error = []
        pts_dt = self.extraer_coordenadas_poligonos(dt_poly)
        pts_gt = self.extraer_coordenadas_poligonos(gt_poly)
        for alpha in np.arange(0, 360, 360 / self.Nr):
            pts1_angulo = self.ptos_con_angulo_alpha(alpha, pts_dt)
            if len(pts1_angulo) == 0:
                continue
            pts1_media = pts1_angulo[0]
            pts2_angulo = self.ptos_con_angulo_alpha(alpha, pts_gt)
            if len(pts2_angulo) == 0:
                continue
            pts2_media = pts2_angulo[0]
            # 2.0 busco punto intermedio (media)
            radio_dt = self.calcular_radio(pts1_media)
            radio_gt = self.calcular_radio(pts2_media)
            diferencia = radio_dt - radio_gt
            error.append(diferencia)

        return np.sqrt((np.array(error)**2).mean())

    def graficar_mapa_color(self):
        polares_heat_map = np.zeros((len(self.gt_poly),self.Nr)) - INFINITO
        for idx,gt in enumerate(self.gt_poly):
            pts = self.extraer_coordenadas_poligonos(gt)
            posiciones = self.rayos_img[pts[:,1],pts[:,0]]-1
            polares_heat_map[idx,posiciones] = self.mapa_color[pts[:,1],pts[:,0]]


        #polares
        #https://stackoverflow.com/questions/36513312/polar-heatmaps-in-python




        #polares_heat_map[mask] = np.nan

        rad = np.linspace(0, len(self.gt_poly),len(self.gt_poly))
        theta = np.linspace(0, 2 * np.pi, self.Nr )
        th, r = np.meshgrid(theta,rad)
        z = polares_heat_map.copy()#np.ma.array(polares_heat_map,mask=[polares_heat_map==np.nan])


        #cmaps = ['inferno','hot','plasma','magma','Blues']
        cmaps = ['Spectral','RdYlGn']
        for cmap_label in cmaps:
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='polar')
            pcm = ax.pcolormesh(th,r,z,cmap=plt.get_cmap(cmap_label))
            #plt.grid()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_theta_zero_location('S')
            fig.colorbar(pcm, ax=ax, orientation="vertical")
            fig.savefig(f"{self.output_dir}/mapa_color_{cmap_label}.png")
            #plt.show()
            #plt.savefig(f"{self.output_dir}/mapa_color_{cmap_label}.png")
            plt.close()

    def calcular_radio(self,pt):
        radio = np.sqrt((self.centro[0] - pt[1]) ** 2 + (self.centro[1] - pt[0]) ** 2)
        return radio

    def calcular_rmse_global(self):
        #x,y = np.where(self.mapa_color.isnan())
        mask = ~np.isnan(self.mapa_color)
        overall_rmse = np.sqrt((self.mapa_color[mask]**2).mean())
        return overall_rmse

    def calcular_mse_por_gt(self):
        self.list_mse = []
        for idx_gt,gt in enumerate(self.gt_poly):
            if idx_gt not in self.asignacion_entre_dt_y_gt:
                self.list_mse.append(0)
                continue
            x, y = gt.exterior.coords.xy
            error = self.mapa_color[np.array(y).astype(int),np.array(x).astype(int)]
            mask = ~np.isnan(error)
            error = error[mask]
            mse = (error**2).mean()
            self.list_mse.append(np.sqrt(mse))


        plt.figure()
        plt.bar(np.arange(0,len(self.list_mse)),self.list_mse)
        plt.title(f"RMSE global={self.calcular_rmse_global():.3f}")
        plt.xlabel('Numero Ring')
        plt.ylabel('RMSE (por gt)')
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/mse.png")
        #plt.show()
        plt.close()


    def ptos_con_angulo_alpha(self,alpha,pts):
        # x,y = pts[:,0], pts[:,1]
        # mask = np.where(self.rayos_img[y,x]==alpha)[0]
        # if len(mask) >= 3:
        #     plt.figure()
        #     plt.imshow(self.img)
        #     plt.scatter(x,y)
        #     plt.figure()
        #     plt.plot(self.rayos_img[y,x])
        #     #plt.show()
        # assert len(mask)<3
        return pts[int(alpha)].reshape(1,2)

    def polinomio_intermedio(self,pol1,pol2,Nr):
        """pol1 incluido en pol2"""
        pts1 = self.extraer_coordenadas_poligonos(pol1)
        pts2 = self.extraer_coordenadas_poligonos(pol2)

        puntos_nuevo_poligono = []
        for alpha in np.arange(0,360,360/Nr):
            #1.0 busco todos los puntos que tienen el mismo angulo, basado en la matriz de angulos

            pts1_angulo = self.ptos_con_angulo_alpha(alpha,pts1)
            if len(pts1_angulo)==0:
                continue
            pts1_media = pts1_angulo.mean(axis=0)
            pts2_angulo = self.ptos_con_angulo_alpha(alpha, pts2)
            if len(pts2_angulo)==0:
                continue
            pts2_media = pts2_angulo.mean(axis=0)
            #2.0 busco punto intermedio (media)
            pt_medio = (0.5*(pts2_media + pts1_media)).astype(int)
            #3.0 agrego pto medio a poligono list
            puntos_nuevo_poligono.append(pt_medio)
        polygon = Polygon(puntos_nuevo_poligono)
        return polygon

    def _asignar_areas_de_influencia(self,img,gt_poly):
        matriz_influencia = np.zeros((img.shape[0],img.shape[1]))-1
        gt_poly.sort(key=lambda x: x.area)

        #inicializacion para deliminar regiones
        # from matplotlib.path import Path
        M, N, _ = self.img.shape
        i = 0
        for gt_i in gt_poly:
            gt_i_plus_1 = gt_poly[i+1] if i < len(gt_poly)-1 else None
            gt_i_minus_1 = gt_poly[i-1] if i > 0 else None

            Cm = self.polinomio_intermedio(gt_i,gt_i_minus_1, self.Nr) if gt_i_minus_1 is not None else None
            CM = self.polinomio_intermedio(gt_i, gt_i_plus_1,self.Nr) if gt_i_plus_1 is not None else None

            #self._plot_gt_and_dt_polys(self.img,Cm,CM)
            #TODO  a todos los pixeles delimitados por los poligonos Cm y CM les asigno el valor i
            if CM is None:
                puntos_nuevo_poligono = []
                pts_cm = self.extraer_coordenadas_poligonos(Cm)
                pts_i = self.extraer_coordenadas_poligonos(gt_i)
                for alpha in np.arange(0,360,360/self.Nr):
                    ptsi_angulo = self.ptos_con_angulo_alpha(alpha, pts_i)
                    if len(ptsi_angulo) == 0:
                        continue
                    pts_i_media = ptsi_angulo.mean(axis=0)
                    pts_cm_angulo = self.ptos_con_angulo_alpha(alpha, pts_cm)
                    if len(pts_cm_angulo) == 0:
                        continue
                    pts_cm_media = pts_cm_angulo.mean(axis=0)
                    # 2.0 busco punto intermedio (media)
                    pt_medio = (pts_i_media + (pts_i_media-pts_cm_media)).astype(int)
                    # 3.0 agrego pto medio a poligono list
                    puntos_nuevo_poligono.append(pt_medio)
                CM = Polygon(puntos_nuevo_poligono)

            if Cm is None:
                contours = self.extraer_coordenadas_poligonos(CM)
                mask = np.zeros((M, N))
                cv.fillPoly(mask, pts=[contours], color=(255))


            else:
                contours = self.extraer_coordenadas_poligonos(CM)
                mask_M = np.zeros((M,N))
                cv.fillPoly(mask_M, pts=[contours], color=(255))

                contours = self.extraer_coordenadas_poligonos(Cm)
                mask_m = np.zeros((M, N))
                cv.fillPoly(mask_m, pts=[contours], color=(255))
                mask = mask_M - mask_m


            #plt.figure();plt.imshow(mask);plt.show();plt.close();
            matriz_influencia[mask>0] = i
            i+=1

        return matriz_influencia





    def true_positive(self):
        return len(self.asignacion_entre_dt_y_gt)


    def false_negative(self):
        gt_ids = np.arange(0,len(self.gt_poly))
        intersect = np.intersect1d(gt_ids,self.asignacion_entre_dt_y_gt)
        return len(self.gt_poly) - len(intersect)


    def false_positive(self):
        return np.where(np.array(self.asignacion_entre_dt_y_gt)==FP_ID)[0].shape[0]

    def true_negative(self):
        return 0


    def precision(self,TP,FP,TN,FN):
        return TP / (TP+FP) if TP+FP>0 else 0

    def recall(self, TP, FP, TN, FN):
        return TP / (TP+FN) if TP+FN>0 else 0

    def fscore(self,TP,FP,TN,FN):
        P = self.precision(TP,FP,TN,FN)
        R = self.recall(TP,FP,TN,FN)
        return 2*P*R / (P+R) if P+R>0 else 0


    def _asignar_ground_truth_a_detecciones(self,matriz_influencias, dt_poly):
        self.asignacion_entre_dt_y_gt = []
        self.porcentaje_de_acierto = []
        for poly in dt_poly:
            x,y = poly.exterior.coords.xy
            x = np.array(x).astype(int)
            y = np.array(y).astype(int)
            gts = matriz_influencias[y,x].astype(int)
            mask_no_background = np.where(gts>=0)[0]
            counts = np.bincount(gts[mask_no_background])

            if len(counts)==0:
                self.asignacion_entre_dt_y_gt.append(FP_ID)
                continue

            gt_label = counts.argmax()
            if gt_label in self.asignacion_entre_dt_y_gt:
                rmse_current = self.calcular_rmse_entre_dt_y_gt(poly,self.gt_poly[gt_label])
                id_dt_former = np.where(self.asignacion_entre_dt_y_gt==gt_label)[0][0]
                rmse_former = self.calcular_rmse_entre_dt_y_gt(self.dt_poly[id_dt_former], self.gt_poly[gt_label])
                if rmse_current < rmse_former:
                    self.asignacion_entre_dt_y_gt[id_dt_former] = FP_ID
                    try:
                        self.porcentaje_de_acierto.pop(id_dt_former)
                    except Exception as e:
                        continue
                else:
                    self.asignacion_entre_dt_y_gt.append(FP_ID)
                    continue

            self.asignacion_entre_dt_y_gt.append(gt_label)
            self.porcentaje_de_acierto.append(counts[gt_label] / counts.sum())
            x_gt,y_gt = self.gt_poly[gt_label].exterior.coords.xy
            #print(f"dt_count = {len(y)}. gt_count={len(y_gt)}")

    def _graficar_asignacion_detecciones_a_gt(self):
        import itertools
        import matplotlib.cm as cm
        M,N,_ = self.img.shape
        plt.figure(figsize=(15,15))
        plt.imshow(np.zeros((M,N)), cmap='gray')

        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        for idx,dt in enumerate(self.dt_poly):
            c = next(colors)
            x,y = dt.exterior.coords.xy
            plt.plot(x, y, color=c)
            #plt.scatter(x,y,s=1)
            if len(self.asignacion_entre_dt_y_gt)-1 < idx:
                continue
            gt_idx = self.asignacion_entre_dt_y_gt[idx]
            if gt_idx == FP_ID:
                plt.plot(x, y, color='w')
                continue
            x,y = self.gt_poly[gt_idx].exterior.coords.xy
            plt.plot(x, y,  color=c)
            #plt.scatter( x,y, s=1)

        plt.axis('off')
        plt.savefig(f'{self.output_dir}/asignacion_dt_gt.png')
        plt.close()

    def computar_indicador_por_deteccion(self):
        #print(f"Porcentaje de Acierto Por Deteccion: {self.porcentaje_de_acierto}")
        nothing = True

    def compute_indicators(self,debug=False):
        matriz_influencias = self._asignar_areas_de_influencia(self.img,self.gt_poly)
        self._graficar_matriz_influencias(matriz_influencias,self.gt_poly)
        self._asignar_ground_truth_a_detecciones(matriz_influencias,self.dt_poly)
        self._graficar_asignacion_detecciones_a_gt()
        self.computar_indicador_por_deteccion()
        TP = self.true_positive()
        FP = self.false_positive()
        TN = self.true_negative()
        FN = self.false_negative()
        self._plot_gt_and_dt_polys(self.img, self.gt_poly, self.dt_poly, n=3,title=f"TP={TP} FP={FP} TN={TN} FN={FN}")
        return TP, FP, TN, FN

    def _graficar_matriz_influencias(self,matriz,list_gt_poly):
        import itertools
        import matplotlib.cm as cm
        M,N,_ = self.img.shape
        plt.figure(figsize=(15, 15))
        plt.imshow(np.zeros((M,N)), cmap='gray')
        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        regiones_id = list(np.unique(matriz).astype(int))
        regiones_id.remove(-1)

        for region in regiones_id:
            y, x = np.where(matriz == region)
            plt.scatter(x, y, s=1, color=next(colors))

        for poly in list_gt_poly:
            x,y = poly.exterior.coords.xy
            plt.plot(x, y,'k')


        plt.axis('off')
        plt.savefig(f'{self.output_dir}/areas_influencia.png')
        plt.close()


def extraerPixelesPertenecientesAlPerfil(copia, angle, centro=None):
    """
        angulo =  {0,pi/4,pi/2,3pi/4,pi,5pi/4,6pi/4,7pi/4}
        ptosCard= {S, SE , E  , NE  , N, NW  , W   , SW   }
         |
        ----------->x
         |
         | IMAGEN
         |
         y

    """
    i = 0
    M, N = copia.shape
    y_pix = []
    x_pix = []
    angle_rad = angle * np.pi / 180
    ctrl = True
    while ctrl:
        x = centro[1] + i * np.sin(angle_rad)
        y = centro[0] + i * np.cos(angle_rad)
        x = x.astype(int)
        y = y.astype(int)

        # print(f'y={y} x={x}')

        if i == 0 or not (x == x_pix[-1] and y == y_pix[-1]):
            y_pix.append(y)
            x_pix.append(x)
        if y >= M - 1 or y <= 1 or x >= N - 1 or x <= 1:
            ctrl = False

        i += 1

    return np.array(y_pix), np.array(x_pix)


def build_radial_directions_matrix(img_seg, centro, Nr):
    perfils_matrix = np.zeros_like(img_seg)
    rango = np.arange(0, 360, 360 / Nr)
    perfiles = {}
    for angulo_perfil in rango:
        y, x = extraerPixelesPertenecientesAlPerfil(
            img_seg, angulo_perfil, [centro[1], centro[0]]
        )

        yy, xx = (
            y.reshape(-1, 1),
            x.reshape(-1, 1)
        )
        perfiles[angulo_perfil] = np.hstack((yy, xx))
        perfils_matrix[yy, xx] = 1 + angulo_perfil

    return perfils_matrix


class ActiveCoutours:
    def __init__(self, gt, dt):
        self.gt_path = Path(gt)
        image_name = self.gt_path.parts[-1][:-5]
        results_dir = get_path('results')
        self.detections_path = dt#results_dir / version / image_name / "labelme.json"
        self.img,centro = load_image(image_name)
        M,N,_ = self.img.shape
        self.radial_matrix = build_radial_directions_matrix(np.zeros((M,N)).astype(np.uint8),centro)
        #kernel = np.ones((3,3), 'uint8')
        #self.radial_matrix = cv.dilate(self.radial_matrix, kernel, iterations=1)
        self.gt_poly = self.load_ring_stimation(self.gt_path)
        self.dt_poly = self.load_ring_stimation(self.detections_path)
        self.gt = self.apply_math_operator_to_each_ring(1, self.gt_poly)
        self.s = self.apply_math_operator_to_each_ring(1, self.dt_poly)



    def load_ring_stimation(self,path):
        json_content = load_json(path)
        anillos = {}
        for ring in json_content['shapes']:
            x,y = Polygon(ring['points']).exterior.coords.xy
            anillos[ring['label']] = np.vstack((x, y)).T.astype(np.int32)

        return anillos

    def dilatar_polinomio(self,poly,thickness,color = (255, 0, 0)):
        isClosed = True
        M,N,_ = self.img.shape
        img = np.zeros((M,N))
        pts = poly
        pts = pts.reshape((-1, 1, 2))
        img = cv.polylines(img, [pts], isClosed, color, thickness)
        x, y = np.where(img > 0)
        vector = np.vstack((x, y)).T.astype(np.int32)
        return vector

    def apply_math_operator_to_each_ring(self, n, dictionary):
        dilated = {}
        for key in dictionary.keys():
            dilated[key] = self.dilatar_polinomio(dictionary[key], n)

        return dilated
    def draw_polys(self,dictionary,n,color,img):
        isClosed = True
        thickness = n
        for key in dictionary.keys():
            pts = dictionary[key].reshape((-1, 1, 2))
            img|= cv.polylines(img, [pts],
                                isClosed, color,
                                thickness)
        return img

    def show_image(self,s,n_s,gt,n_gt,title):
        img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        img |= self.draw_polys(gt, n_gt, (0, 255, 0), img)
        img |= self.draw_polys(s,n_s,(255,0,0),img)

        plt.figure(figsize=(10,10));plt.title(title);plt.imshow(img);plt.axis('off');#plt.show()


    def true_positive_and_false_negative(self,s,gt_d,dt_poly,gt_poly,threshold):
        TP = 0
        M,N,_ = self.img.shape
        img_gt = np.zeros((M,N)).astype(np.uint8)
        for key in gt_d.keys():
            x,y = gt_d[key][:,0],gt_d[key][:,1]
            img_gt[x,y] = int(key)


        for key in s.keys():
            img_s = np.zeros_like(img_gt)
            x, y = s[key][:, 0], s[key][:, 1]
            img_s[x, y] = 255
            s_inter_gtd = img_gt & img_s & self.radial_matrix

            total_puntos_anillo_dt = self.contar_intersecciones(img_s & self.radial_matrix)
            total_intersecciones = self.contar_intersecciones(s_inter_gtd)
            if total_intersecciones > threshold * total_puntos_anillo_dt:
                TP+=1
                num_labels, interseccion = cv.connectedComponents(img_gt & img_s, connectivity=8)
                x,y = np.where(interseccion>0)
                key_gt = str(np.unique(img_gt[x,y])[0])
                x, y = gt_d[key_gt][:, 0], gt_d[key_gt][:, 1]
                img_gt[x, y] = 0

        num_labels, interseccion = cv.connectedComponents(img_gt, connectivity=8)
        FN = num_labels-1
        return TP,FN

    def true_positive(self,s,gt_d,dt_poly,gt_poly,threshold):
        TP = 0
        M,N,_ = self.img.shape
        img_gt = np.zeros((M,N)).astype(np.uint8)
        for key in gt_d.keys():
            x,y = gt_d[key][:,0],gt_d[key][:,1]
            img_gt[x,y] = 255


        for key in s.keys():
            img_s = np.zeros_like(img_gt)
            x, y = s[key][:, 0], s[key][:, 1]
            img_s[x, y] = 255
            s_inter_gtd = img_gt & img_s & self.radial_matrix

            total_puntos_anillo_dt = self.contar_intersecciones(img_s & self.radial_matrix)
            total_intersecciones = self.contar_intersecciones(s_inter_gtd)
            if total_intersecciones > threshold * total_puntos_anillo_dt:
                TP+=1


        return TP

    def contar_intersecciones(self,matriz):
        num_labels, _ = cv.connectedComponents(matriz, connectivity=8)
        return num_labels - 1

    def false_negative(self,s_d,gt,dt_poly,gt_poly,threshold):
        #TODO: no esta bien implementado. Mirar funcion true_positive_and_false_negative.
        M, N, _ = self.img.shape
        FN = 0
        img_sd_c = np.ones((M,N)).astype(np.uint8)*255
        for idx,key in enumerate(s_d.keys()):
            x, y = s_d[key][:, 0], s_d[key][:, 1]
            img_sd_c[x, y] = 0

        for key in gt.keys():
            img_gt = np.zeros((M,N)).astype(np.uint8)
            x, y = gt[key][:, 0], gt[key][:, 1]
            img_gt[x, y] = 255
            s_inter_gt = img_gt & img_sd_c & self.radial_matrix
            total_puntos_anillo_gt = self.contar_intersecciones( img_gt & self.radial_matrix)
            total_intersecciones = self.contar_intersecciones(s_inter_gt)
            if total_intersecciones > threshold * total_puntos_anillo_gt:
                FN += 1
        return FN


    def false_positive(self,s,TP):
        return len(s.keys()) - TP

    def true_negative(self,s_d,gt_d,threshold):
        #TODO: no esta bien implementado. Pero no se usa en fscore
        TN = 0
        M, N, _ = self.img.shape
        img_gt = np.zeros((M,N)).astype(np.uint8)
        for key in gt_d.keys():
            x, y = gt_d[key][:, 0], gt_d[key][:, 1]
            img_gt[x, y] = 255

        for key in s_d.keys():
            img_s = np.zeros((M,N)).astype(np.uint8)
            x, y = s_d[key][:, 0], s_d[key][:, 1]
            img_s[x, y] = 255
            s_inter_gtd = img_gt & img_s & self.radial_matrix
            total_intersecciones = self.contar_intersecciones(s_inter_gtd)
            total_puntos = self.contar_intersecciones(img_s & self.radial_matrix)
            if total_intersecciones > threshold * total_puntos:
                TN+=1

        return TN


    def precision(self,TP,FP,TN,FN):
        return TP / (TP+FP) if TP+FP>0 else 0

    def recall(self, TP, FP, TN, FN):
        return TP / (TP+FN) if TP+FN>0 else 0

    def fscore(self,TP,FP,TN,FN):
        P = self.precision(TP,FP,TN,FN)
        R = self.recall(TP,FP,TN,FN)
        return 2*P*R / (P+R) if P+R>0 else 0

    def compute_threshold(self,n=5,threshold=0.75,debug=False,max_error=10):
        fvalues = []
        print(f"\tth \tTP \tFP \tTN \tFN \t\tP \t\tR \t\tF @n={n}")
        for threshold in np.arange(0.1,1,1/max_error):
            TP, FP, TN, FN = self.compute_indicators(n, threshold)
            # 3.0 compute metrics
            F = self.fscore(TP, FP, TN, FN)
            P = self.precision(TP, FP, TN, FN)
            R = self.recall(TP, FP, TN, FN)
            print(f"\t{threshold:.2f}\t{TP}\t{FP}\t{TN}\t{FN}\t{P:.2f}\t{R:.2f}\t{F:.2f}")
            fvalues.append(F)

        self.graficar_fscore_threshold(fvalues, np.arange(0.1,1,1/max_error), n)
        return TP, FN, TN, FP

    def compute_indicators(self,n,threshold,debug=False):
        gt_d = self.apply_math_operator_to_each_ring(n, self.gt_poly)
        s_d = self.apply_math_operator_to_each_ring(n, self.dt_poly)

        # 2.0 compute indicators
        #TP = self.true_positive(self.s, gt_d,self.dt_poly,self.gt_poly, threshold)
        TP,FN = self.true_positive_and_false_negative(self.s, gt_d,self.dt_poly,self.gt_poly, threshold)
        FP = self.false_positive(self.s, TP)
        TN = self.true_negative(s_d, gt_d, threshold)
        #FN = self.false_negative(s_d, self.gt, self.dt_poly,self.gt_poly, threshold)

        return TP,FP,TN,FN

    def compute(self,threshold=0.75,debug=False,max_error=16):
        fvalues = []
        self.show_image(self.dt_poly, 1, self.gt_poly, 1, f"")
        print(f"\tn \tTP \tFP \tTN \tFN \t\tP \t\tR \t\tF @threshold={threshold}")

        self.show_image(self.dt_poly, 5, self.gt_poly, 5, f"")
        for n in range(1,max_error):
            TP,FP,TN,FN = self.compute_indicators(n,threshold)
            # 3.0 compute metrics
            F = self.fscore(TP,FP,TN,FN)
            P = self.precision(TP,FP,TN,FN)
            R = self.recall(TP,FP,TN,FN)
            print(f"\t{n}\t{TP}\t{FP}\t{TN}\t{FN}\t{P:.2f}\t{R:.2f}\t{F:.2f}")
            fvalues.append(F)

        self.graficar_fscore(fvalues,max_error,threshold)

        return TP, FN, TN, FP

    def graficar_fscore_threshold(self,fvalues,x,n):
        plt.figure()
        plt.plot(x,fvalues)
        plt.ylabel('fscore')
        plt.xlabel('threshold)')
        plt.title(f"n:{n}")
        plt.grid(True)
        plt.show()

    def graficar_fscore(self,fvalues,max_error,threshold):
        plt.figure()
        plt.plot(np.arange(1,max_error),fvalues)
        plt.ylabel('fscore')
        plt.xlabel('margin-error (n)')
        plt.title(f"umbral:{threshold}")
        plt.grid(True)
        plt.show()


class MetricsDataset:
    def __init__(self,root_dir, creador, config_path, output_dir):
        data_path = get_path("data")
        dataset = pd.read_csv(f"{data_path}/dataset.csv")
        self.creator = creador
        self.results_path = get_path('results')
        self.gt_dir = data_path / f"ground_truth/{creador}/labelme"
        self.data = dataset
        self.root_dir = root_dir
        self.table = pd.DataFrame(columns=['imagen', 'TP','FP','TN','FN','P','R','F'] )
        config = load_json(config_path)
        self.Nr = config.get("Nr")
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

    def compute(self):
        for index in tqdm(range(self.data.shape[0]),desc='Computing metrics'):
            disco = self.data.iloc[index]
            imagenNombre = disco["Imagen"]
            dt_dir = Path(f"{self.root_dir}/{imagenNombre[:-4]}")
            dt_file =dt_dir / "labelme.json"
            if (not dt_file.exists()):
                continue
            gt_file = self.gt_dir / f"{imagenNombre[:-4]}.json"
            #(self.results_path / f"v{self.version}/{imagenNombre[:-4]}").mkdir(exist_ok=True)
            #if (not dt_file.exists() or not gt_file.exists()):
            if (not gt_file.exists()):
                row = {'imagen': imagenNombre[:-4],'TP': None,'FP': None,'TN': None,'FN': None,'P': None,'R': None,'F': None}

            else:
                #metrics = ActiveCoutours(gt_file, dt_file)
                img_dir = Path(self.output_dir) / f"{imagenNombre[:-4]}"
                (img_dir).mkdir(exist_ok=True)
                metrics = AreaInfluencia( gt_file, dt_file, str(img_dir), self.Nr)
                TP, FP, TN, FN = metrics.compute_indicators()

                # 3.0 compute metrics
                F = metrics.fscore(TP, FP, TN, FN)
                P = metrics.precision(TP, FP, TN, FN)
                R = metrics.recall(TP, FP, TN, FN)
                metrics.calcular_error_respecto_a_gt()
                metrics.calcular_mse_por_gt()
                RMSE = metrics.calcular_rmse_global()
                row = {'imagen': imagenNombre[:-4],'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'P': P, 'R': R, 'F': F,'RMSE':RMSE}

                self.table = self.table.append(row, ignore_index=True)


        #Add average
        row = {'imagen': "Average", 'TP': -1, 'FP': -1, 'TN': -1, 'FN': -1, 'P': self.table.P.values.mean(),
               'R': self.table.R.values.mean(), 'F': self.table.F.values.mean(),
               'RMSE': self.table.RMSE.values.mean()}
        self.table = self.table.append(row, ignore_index=True)

        self.table = self.table.set_index('imagen')
        return None

    def print_results(self):
        print(self.table)
        return None

    def save_results(self):
        self.table.to_csv(f"{self.output_dir}/results_{self.creator}.csv", sep=',', float_format='%.3f')


class Metrics:
    def __init__(self,versiones_list):
        data_path = get_path("data")
        dataset = pd.read_csv(f"{data_path}/dataset.csv")
        self.results_path =  get_path('results')
        self.data = dataset
        self.versiones = versiones_list
        columns = []
        for ver in versiones_list:
            columns.append(f"{ver}(num)")
            columns.append(f"{ver}(%)")
            
        self.table = pd.DataFrame(columns=['imagen','gt'] + columns)
    
    def compute(self):
        for index, disco in self.data.iterrows():
            imagenNombre = disco["Imagen"]
            row = {'imagen': imagenNombre[:-4], 'gt':disco.gt_cantidad_anillos}
            for version in list(self.versiones):
                result_dir = f"{self.results_path}/v{version}/{imagenNombre[:-4]}/results.json"
                try:
                    res = load_json(result_dir)
                except:
                    continue

                row[f"{version}(num)"] = res['union_cadenas']['anillos']
                row[f"{version}(%)"] = res['union_cadenas']['anillos'] / row['gt']
            self.table = self.table.append(row,ignore_index=True)

        self.table = self.table.set_index('imagen')

        return None
    
    def print_results(self):
        print(self.table)
        return None        
    
    def save_results(self):
        self.table.to_csv(f"{self.root_dir}/results.csv",sep=',')

def main(rood_dir,creador, config_path, output):
    metrics = MetricsDataset(rood_dir,creador, config_path, output)
    metrics.compute()
    metrics.print_results()
    metrics.save_results()

    return



import matplotlib.pyplot as plt
from matplotlib.path import Path as Path_plt
from matplotlib.patches import PathPatch
def get_error_band(ax, x, y, err_x, err_y, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err_x
    yp = y + ny * err_y
    xn = x - nx * err_x
    yn = y - ny * err_y

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T

    codes = np.full(len(vertices), Path_plt.LINETO)
    codes[0] = codes[len(xp)] = Path_plt.MOVETO
    path = Path_plt(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


from lib.drawing import Drawing, Color
from fpdf import FPDF

def compute_mean_gt():
    gt_proccessed_path = '/data/maestria/database_new_nomenclature/ground_truth_processed/annotations'
    dataset = pd.read_csv(f"/data/maestria/database_new_nomenclature/dataset_ipol.csv")
    data = np.zeros((dataset.shape[0], 6))
    etiquetadores = {}
    for idx, row in tqdm(dataset.iterrows()):
        disk_name = row.Imagen
        if 'fx' in disk_name:
            continue

        # if disk_name not in ['F07c']:
        #     continue

        creator_list = []
        for creator in [ 'maria', 'veronica', 'serrana', 'christine']:
            file_gt = f'{gt_proccessed_path}/{disk_name}_{creator}.json'
            if Path(file_gt).exists():
                creator_list.append(file_gt)
            else:
                creator_list.append(None)

        ################################################################################################################
        image_name = f'/data/maestria/database_new_nomenclature/images/{disk_name}.png'
        print(image_name)
        img_array = cv.imread(image_name)
        height, width, _ = img_array.shape
        ################################################################################################################

        gt_creators = []
        Nr = 360
        centro = [row.cy, row.cx]
        rays_list = build_rays(Nr,  width,height, centro)
        for idx_creator, creator_file in enumerate(creator_list):
            if creator_file is None:
                continue
            helper = AreaInfluencia(creator_file, '', '', Nr, disk_name)
            creator_points = helper.load_ring_stimation(creator_file)
            creator_gt = [helper._sampling_poly(poly, row.cy, row.cx, rays_list) for poly in creator_points]
            data[idx, idx_creator] = len(creator_gt)
            gt_creators.append(creator_gt)

        # MeanDisk.draw_ray_curve_and_intersections(rays_list, creator_gt,
        #                                           [], [], img_array, './debug.png')
        ################################################################################################################
        #computar gt medio
        output_filename = f'{gt_proccessed_path}/{disk_name}_mean.json'
        mean_disk = MeanDisk( gt_creators, height, width, row.cy, row.cx, image_name, output_filename, img_array)
        data[idx,[4,5]] = [height, width]


    np.savetxt(f'{gt_proccessed_path}/comparison/data.txt',data)


class MeanDisk:
    def __init__(self, annotations_lists, height, width, cy, cx, image_path, output_filename, img, nr=360):
        # change grouping of annotations. From creators gt to ring gt.
        self.img = img
        disk_rings_list = self.grouping_gt_annotations(annotations_lists)

        #compute mean ring
        mean_ring_gt = self.compute_mean_gt(nr, height, width, cy, cx, disk_rings_list)
        ############################################################################################################
        #save gt to labelme
        self.save_gt_to_labelme(mean_ring_gt, image_path, height, width, output_filename=output_filename)
        self.mean_ring_gt = mean_ring_gt

    @staticmethod
    def grouping_gt_annotations(annotations_lists):
        disk_rings_list = []
        total_ring_per_disk = len(annotations_lists[0])
        #sort disk
        for idx in range(len(annotations_lists)):
            annotations_lists[idx].sort(key = lambda x: x.area)

        for idx_ring in range(total_ring_per_disk):
            rings_list = []
            for creator_gt in annotations_lists:
                rings_list.append(creator_gt[idx_ring])

            disk_rings_list.append(rings_list)

        return disk_rings_list

    @staticmethod
    def draw_ray_curve_and_intersections(rays_list, poly_list, other_poly_list, intersection_list, img, filename):
        img_draw = img.copy()
        for ray in rays_list:
            img_draw = Drawing.radii(ray, img_draw)

        for poly in poly_list:
            img_draw = Drawing.curve(poly.exterior.coords, img_draw)

        for poly in other_poly_list:
            img_draw = Drawing.curve(poly.exterior.coords, img_draw, color=Color.red)

        for node in intersection_list:
            img_draw = Drawing.intersection(node, img_draw, color=Color.yellow)

        cv.imwrite(filename, img_draw)


    def compute_mean_gt(self, nr, height, width, cy, cx, disk_rings_list):
        mean_ring_gt = []
        rays_list = build_rays(nr, height, width, [cy, cx])

        intersection_list = {}
        poly_list = []
        total_ring_per_disk = len(disk_rings_list)
        for idx_ring in range(total_ring_per_disk):
            intersection_list[idx_ring] = []
            for creator_gt in disk_rings_list[idx_ring]:
                intersection_list[idx_ring] += creator_gt.node_list
                poly_list.append(creator_gt)

            ############################################################################################################
            mean_ring_point = []
            for direction in np.arange( 0, nr):
                node_in_direction = [node for node in intersection_list[idx_ring] if node.angle == direction]
                x_mean = np.mean([node.x for node in node_in_direction])
                y_mean = np.mean([node.y for node in node_in_direction])
                mean_ring_point.append([y_mean, x_mean])
            ############################################################################################################
            mean_ring_gt.append(Polygon(mean_ring_point))




        self.intersection_full_list = [node for key in intersection_list.keys() for node in intersection_list[key]]
        params = {'y': cy, 'x': cx, 'angle': int(0), 'radial_distance':
            0, 'chain_id': -1}

        dot = ch.Node(**params)
        self.intersection_full_list.append(dot)
        #self.draw_ray_curve_and_intersections( rays_list, mean_ring_gt,poly_list,self.intersection_full_list ,self.img, './debug_1.png')

        return mean_ring_gt

    @staticmethod
    def save_gt_to_labelme(mean_ring_gt, image_path, height, width, output_filename):
        labelme_json = {"imagePath": image_path, "imageHeight": height,
                        "imageWidth": width, "version": "5.0.1",
                        "flags": {}, "shapes": [], "imageData": None}

        for idx, mean_ring in enumerate(mean_ring_gt):
            anillo = {"label": str(idx + 1)}
            y, x = mean_ring.exterior.coords.xy
            anillo["points"] = [[i,j] for i,j in zip(y,x)]
            anillo["shape_type"] = "polygon"
            anillo["flags"] = {}
            labelme_json["shapes"].append(anillo)

        write_json(labelme_json, output_filename)

        return




    @staticmethod
    def compute_intersections( gt_ring, rays_list, cy, cx):
        center = [cy,cx]
        chain_id = -1
        intersection_list = []
        for radii in rays_list:
            try:
                inter = radii.intersection(gt_ring)
            except Exception:
                from shapely.validation import make_valid
                valid_shape = make_valid(gt_ring)
                inter = radii.intersection(valid_shape)

            if not inter.is_empty:
                if 'MULTIPOINT' in inter.wkt:
                    inter = inter[0]

                if type(inter) == Point:
                    y, x = inter.xy

                elif type(inter) == LineString:
                    y, x = inter.coords.xy

                elif 'MULTILINESTRING' in inter.wkt:
                    y, x = inter[0].coords.xy

                else:
                    continue

                i, j = np.array(y)[-1], np.array(x)[-1]
                params = {'y': i, 'x': j, 'angle': int(radii.direction), 'radial_distance':
                    ch.euclidean_distance([i, j], center), 'chain_id': chain_id}

                dot = ch.Node(**params)
                intersection_list.append(dot)

            else:
                raise

        return intersection_list


def draw_creator_annotation_over_disk():
    dataset = pd.read_csv(f"/data/maestria/database_new_nomenclature/dataset_ipol.csv")
    gt_processed_root_path = Path('/data/maestria/database_new_nomenclature/ground_truth_processed')
    gt_processed_root_path.mkdir(exist_ok=True)

    gt_processed_path = Path(f'{gt_processed_root_path}/annotations')
    gt_processed_path.mkdir(exist_ok=True)

    gt_processed_path_images = Path(f"{gt_processed_root_path}/images")
    gt_processed_path_images.mkdir(exist_ok=True)

    pdf = FPDF()
    pdf.set_font('Arial', 'B', 16)
    x, y = 0, 40
    height_pdf = 140
    for idx, row in tqdm(dataset.iterrows()):
        disk_name = row.Imagen
        if 'fx' in disk_name:
            continue
        # if disk_name not in ['F07c']:
        #     continue
        image_name = f'/data/maestria/database_new_nomenclature/images/{disk_name}.png'
        creator_list = []
        maria = f'{gt_processed_path}/{disk_name}_maria.json'
        if Path(maria).exists():
            creator_list.append(maria)

        veronica = f'{gt_processed_path}/{disk_name}_veronica.json'
        if Path(veronica).exists():
            creator_list.append(veronica)

        serrana = f'{gt_processed_path}/{disk_name}_serrana.json'
        if Path(serrana).exists():
            creator_list.append(serrana)

        christine = f'{gt_processed_path}/{disk_name}_christine.json'
        if Path(christine).exists():
            creator_list.append(christine)

        christine = f'{gt_processed_path}/{disk_name}_mean.json'
        if Path(christine).exists():
            creator_list.append(christine)

        img_array = cv.imread( image_name, cv.COLOR_RGB2GRAY)
        if img_array is None:
            break
        img = np.zeros_like(img_array)
        height, widht, _ = img_array.shape
        img[:, :, 0] = img_array[:, :, 0]
        img[:, :, 1] = img_array[:, :, 1]
        img[:, :, 2] = img_array[:, :, 2]
        centro = [row.cy, row.cx]
        Nr = 360
        rays_list = build_rays(Nr, widht, height, centro)
        color = Color()
        for creator in creator_list:
            helper = AreaInfluencia(creator, '', '', Nr, disk_name)
            creator_points = helper.load_ring_stimation(creator)
            creator_gt = [helper._sampling_poly(poly, row.cy, row.cx, rays_list) for poly in creator_points]
            creator_color = color.get_next_color() if  'mean' not in Path(creator).name else Color.black
            for gt in creator_gt:
                img = Drawing.curve(gt.exterior.coords, img_array, creator_color, thickness = 1)

        output_filename = f'{str(gt_processed_path_images)}/{disk_name}_creators_only.png'
        cv.imwrite( output_filename, img)

        pdf.add_page()
        pdf.cell(0, 0, disk_name)
        pdf.image(output_filename, x, y, h=height_pdf)




    pdf.output(f"{str(gt_processed_root_path)}/postprocessed_gt.pdf", 'F')



    return



def process_gt_files():
    creators = [ 'maria', 'veronica', 'christine', 'serrana']
    data_path = get_path("data")
    dataset = pd.read_csv(f"{data_path}/dataset.csv")
    gts_path = '/data/maestria/datasets/ground_truth/analisis'

    gt_maria = f"{gts_path}/maria/results_maria.csv"
    df_maria = pd.read_csv(gt_maria)

    gt_veronica = f"{gts_path}/veronica/results_veronica.csv"
    df_veronica= pd.read_csv(gt_veronica)

    gt_serrana = f"{gts_path}/serrana/results_serrana.csv"
    df_serrana = pd.read_csv(gt_serrana)

    gt_christine = f"{gts_path}/christine/results_christine.csv"
    df_christine  = pd.read_csv(gt_christine)

    table = pd.DataFrame(columns=['imagen','maria','veronica','serrana','christine'])
    for idx, row in dataset.iterrows():
        christine = maria = veronica = serrana = 0
        disk_name = row.Imagen[:-4]
        if disk_name in df_serrana.imagen.values.tolist():
            serrana = df_serrana[df_serrana.imagen == disk_name].gt_rings.values[0]

        if disk_name in df_maria.imagen.values.tolist():
            maria = df_maria[df_maria.imagen == disk_name].gt_rings.values[0]

        if disk_name in df_veronica.imagen.values.tolist():
            veronica = df_veronica[df_veronica.imagen == disk_name].gt_rings.values[0]

        if disk_name in df_christine.imagen.values.tolist():
            christine = df_christine[df_christine.imagen == disk_name].gt_rings.values[0]

        data = {'imagen': disk_name,'christine':christine,'maria':maria,'serrana': serrana, 'veronica': veronica}
        table = table.append(data, ignore_index=True)
    table.to_csv(f"{gts_path}/gt_processed.csv", sep=',')

    return 0

import os

def cp_gt_files_to_new_location():
    creators = ['maria', 'veronica', 'christine', 'serrana']
    data_path = get_path("data")
    dataset = pd.read_csv(f"{data_path}/dataset.csv")
    for creator in creators:
        gt_files = f"/data/maestria/datasets/ground_truth/{creator}/labelme/"
        for idx, row in dataset.iterrows():
            disk_name = row.Imagen[:-4]
            gt_disk_file_name = Path(gt_files) / f"{disk_name}.json"
            if not gt_disk_file_name.exists():
                continue
            os.system(f"cp {gt_disk_file_name} /data/maestria/datasets/ground_truth/analisis/gt_json/{disk_name}_{creator}.json")




def main(rood_dir,creador, config_path, output):
    metrics = MetricsDataset(rood_dir,creador, config_path, output)
    metrics.compute()
    metrics.print_results()
    metrics.save_results()

    return


class MetricsDataset_comparison:
    def __init__(self,root_dir, creador, config_path, output_dir):
        dataset = pd.read_csv(f"/data/maestria/database_new_nomenclature/dataset_ipol.csv")
        self.creator = creador
        self.results_path = get_path('results')
        self.gt_dir = root_dir
        self.data = dataset
        self.root_dir = root_dir
        self.table = pd.DataFrame(columns=['imagen', 'TP', 'FP', 'TN', 'FN', 'P', 'R', 'F'])
        config = load_json(config_path)
        self.Nr = config.get("Nr")
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

    def compute(self):
        for index in tqdm(range(self.data.shape[0]),desc='Computing metrics'):
            disco = self.data.iloc[index]
            imagenNombre = disco["Imagen"]
            # if imagenNombre  not in ['L02c']:
            #     continue
            print(imagenNombre)
            dt_file = Path(f"{self.root_dir}/{imagenNombre}_{self.creator}.json")
            # if (not dt_file.exists()):
            #     continue
            gt_file = Path(self.gt_dir) / f"{imagenNombre}_mean.json"
            #(self.results_path / f"v{self.version}/{imagenNombre[:-4]}").mkdir(exist_ok=True)
            #if (not dt_file.exists() or not gt_file.exists()):
            if (not gt_file.exists()) or (not dt_file.exists()):
                row = {'imagen': imagenNombre,'TP': None,'FP': None,'TN': None,'FN': None,'P': None,'R': None,'F': None}

            else:
                #metrics = ActiveCoutours(gt_file, dt_file)
                img_dir = Path(self.output_dir) / f"{imagenNombre}"
                (img_dir).mkdir(exist_ok=True)
                metrics = AreaInfluencia( gt_file, dt_file, str(img_dir), self.Nr)
                TP, FP, TN, FN = metrics.compute_indicators()

                # 3.0 compute metrics
                F = metrics.fscore(TP, FP, TN, FN)
                P = metrics.precision(TP, FP, TN, FN)
                R = metrics.recall(TP, FP, TN, FN)
                metrics.calcular_error_respecto_a_gt()
                metrics.calcular_mse_por_gt()
                RMSE = metrics.calcular_rmse_global()
                row = {'imagen': imagenNombre,'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'P': P, 'R': R, 'F': F,
                       'RMSE':RMSE,'exec_time': load_json(str(dt_file))['exec_time(s)']}

            self.table = self.table.append(row, ignore_index=True)


        #Add average
        row = {'imagen': "Average", 'TP': -1, 'FP': -1, 'TN': -1, 'FN': -1, 'P': self.table.P.values.mean(),
               'R': self.table.R.values.mean(), 'F': self.table.F.values.mean(),
               'RMSE': self.table.RMSE.values.mean()}
        self.table = self.table.append(row, ignore_index=True)

        self.table = self.table.set_index('imagen')
        return None

    def print_results(self):
        print(self.table)
        return None

    def save_results(self):
        self.table.to_csv(f"{self.output_dir}/results_{self.creator}.csv", sep=',', float_format='%.3f')

def main_comparison(creator):
    root_dir = "/data/maestria/database_new_nomenclature/ground_truth_processed/annotations"
    output = f"/data/maestria/database_new_nomenclature/ground_truth_processed/annotations/comparison/{creator}"
    config_path = "./config/general.json"
    metrics = MetricsDataset_comparison(root_dir,creator, config_path, output)
    metrics.compute()
    metrics.print_results()
    metrics.save_results()

    return

def mean_gt_generation_and_comparison_between_creator():
    #compute_mean_gt()
    #draw_creator_annotation_over_disk()
    main_comparison('ipol_refactoring')
    # main_comparison('veronica')
    # main_comparison('serrana')
    # main_comparison('christine')

if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # #parser.add_argument("--creator", type=str, required=True)
    # parser.add_argument("--config_path", type=str, required=True)
    # parser.add_argument("--root_dir", type=str, required=True)
    # parser.add_argument("--output", type=str, required=True)
    # args = parser.parse_args()
    # main(str(args.root_dir),'mean',str(args.config_path), str(args.output))
    mean_gt_generation_and_comparison_between_creator()
    #process_gt_files()
    #cp_gt_files_to_new_location()

