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

# import lib.metrics_sampling as sampling
INFINITO = np.nan


def load_image(image_name, disk_name=None):
    img_name = image_name if disk_name is None else disk_name
    data_path = get_path("data")
    dataset = pd.read_csv(f"/data/maestria/database_new_nomenclature/dataset_ipol.csv")
    # nroImagen = [idx for idx,image_file in enumerate(dataset['Imagen'].values) if (image_file[:-4] in img_name) and (img_name in image_file[:-4])][0]
    disco = dataset[dataset.Imagen == img_name]
    # disco = dataset.iloc[int(nroImagen)]
    path = f"/data/maestria/database_new_nomenclature/images"
    imagenNombre = disco["Imagen"]
    centro = tuple([int(disco["cy"]), int(disco["cx"])])
    return cv.imread(str(path + f"/{img_name}.png"), cv.COLOR_RGB2GRAY), centro


class Polygon_node(Polygon):
    def __init__(self, node_list):
        self.angles = [node.angle for node in node_list]
        self.node_list = node_list
        super().__init__([[node.y, node.x] for node in node_list])


FP_ID = -1


class InfluenceArea:
    def __init__(self, gt, dt, output_dir, Nr, threshold=0.60, disk_name=None):
        self.gt_path = Path(gt)
        self.threshold = threshold
        self.output_dir = output_dir
        self.dt_path = dt
        image_name = self.gt_path.parts[-1]
        for prefix in ['mean', 'maria', 'veronica', 'christine', 'serrana']:
            if prefix in image_name:
                image_name = image_name.replace(f"_{prefix}", "")
        self.image_name = image_name[:-5]
        self.img, self.center = load_image(self.image_name, disk_name)
        M, N, _ = self.img.shape
        # self.rayos_img = build_radial_directions_matrix(np.zeros((img_height,width)),self.center, nr).astype(int)
        # nr, height_output, witdh, [cy, cx]
        self.rays_list = build_rays(Nr, M, N, self.center[::-1])
        # draw_ray_curve_and_intersections([],self.rays_list, [], self.im_pre, "./debug_rays.png")
        self.gt_poly = self.load_ring_stimation(self.gt_path)
        gt_poly_samples = []
        cy, cx = self.center
        for poly in self.gt_poly:
            sampled_poly = self._sampling_poly(poly, cy, cx, self.rays_list, self.img)
            if sampled_poly is None:
                continue
            gt_poly_samples.append(sampled_poly)
        self.gt_poly = gt_poly_samples
        #delete pith
        self.gt_poly.sort(key=lambda x: x.area)
        #self.gt_poly = self.gt_poly[1:]

        self.dt_poly = self.load_ring_stimation(self.dt_path)
        dt_poly_samples = []
        for idx, poly in enumerate(self.dt_poly):
            sampled_poly = self._sampling_poly(poly, cy, cx, self.rays_list, self.img)
            if sampled_poly is None:
                continue
            dt_poly_samples.append(sampled_poly)
        self.dt_poly = dt_poly_samples
        self.Nr = Nr

    @staticmethod
    def load_ring_stimation(path):
        try:
            json_content = load_json(path)
            anillos = []
            for ring in json_content['shapes']:
                anillos.append(Polygon(np.array(ring['points'])[:, [1, 0]].tolist()))

        except FileNotFoundError:
            anillos = []

        return anillos

    def _convert_poly_dict_to_poly_list(self, poly_d):
        poly_list = []
        for key in poly_d.keys():
            poly_list.append(poly_d[key])

        return poly_list

    def _add_poly_to_img(self, img, poly, color, thickness=1):
        isClosed = True
        y, x = poly.exterior.coords.xy
        pts = np.vstack((x, y)).T.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        return cv.polylines(img, [pts], isClosed, color, thickness)

    @staticmethod
    def _sampling_poly(poly, cy, cx, rays_list, img=None):
        # MeanDisk.draw_ray_curve_and_intersections( [radii for radii in self.rays_list if radii.direction ==93], [poly], [], [], self.im_pre, './debug.png')

        intersection_list = MeanDisk.compute_intersections(poly, rays_list, cy, cx, img=img)
        if intersection_list is None:
            return None
        intersection_list.sort(key=lambda x: x.angle)
        return Polygon_node(intersection_list)

    def _plot_gt_and_dt_polys(self, img, gt, dt, n=1, title=None):
        img_aux = img.copy()
        # gt
        for poly in gt:
            img_aux = self._add_poly_to_img(img_aux, poly, color=(0, 255, 0), thickness=n)

        # dt
        for poly in dt:
            img_aux = self._add_poly_to_img(img_aux, poly, color=(255, 0, 0), thickness=n)

        plt.figure(figsize=(10, 10));
        plt.imshow(img_aux);
        plt.axis('off');
        if title is not None:
            plt.title(title)
        plt.savefig(f'{self.output_dir}/dt_and_gt.png')
        # plt.show();
        plt.close()

    def extraer_coordenadas_poligonos(self, poly):
        x, y = poly.exterior.coords.xy
        pts = np.vstack((x, y)).T.astype(np.int32)
        return pts

    def compute_error_with_gt(self, infinito=INFINITO):
        self.mapa_color = np.zeros((len(self.gt_poly), 360))
        for idx, dt in enumerate(self.dt_poly):
            pts_dt = self.extraer_coordenadas_poligonos(dt)
            if len(self.dt_and_gt_assignation) - 1 < idx:
                continue

            gt_idx = self.dt_and_gt_assignation[idx]
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
                diferencia = radio_dt - radio_gt
                # self.mapa_color[pts2_media[1],pts2_media[0]] = diferencia
                self.mapa_color[gt_idx, int(alpha)] = diferencia
        self.graficar_mapa_color()
        return 0

    def compute_rmse_between_dt_and_gt(self, dt_poly, gt_poly):
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

        return np.sqrt((np.array(error) ** 2).mean())

    def graficar_mapa_color(self):
        polares_heat_map = self.mapa_color.copy()

        # polares
        # https://stackoverflow.com/questions/36513312/polar-heatmaps-in-python

        # polares_heat_map[mask] = np.nan

        rad = np.linspace(0, len(self.gt_poly), len(self.gt_poly))
        theta = np.linspace(0, 2 * np.pi, self.Nr)
        th, r = np.meshgrid(theta, rad)
        z = polares_heat_map.copy()  # np.ma.array(polares_heat_map,mask=[polares_heat_map==np.nan])

        # cmaps = ['inferno','hot','plasma','magma','Blues']
        cmaps = ['Spectral', 'RdYlGn']
        for cmap_label in cmaps:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')
            pcm = ax.pcolormesh(th, r, z, cmap=plt.get_cmap(cmap_label))
            # plt.grid()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_theta_direction(-1)
            # ax.set_theta_zero_location('S')
            ax.set_theta_offset(np.pi / 2)
            fig.colorbar(pcm, ax=ax, orientation="vertical")
            fig.savefig(f"{self.output_dir}/mapa_color_{cmap_label}.png")
            # plt.show()
            # plt.savefig(f"{self.output_dir}/mapa_color_{cmap_label}.png")
            plt.close()

    def calcular_radio(self, pt):
        radio = np.sqrt((self.center[0] - pt[1]) ** 2 + (self.center[1] - pt[0]) ** 2)
        return radio

    def compute_rmse_global(self):
        # x,y = np.where(self.mapa_color.isnan())
        mask = ~np.isnan(self.mapa_color)
        overall_rmse = np.sqrt((self.mapa_color[mask] ** 2).mean())
        return overall_rmse

    def compute_mse_per_gt(self):
        self.list_mse = []
        for idx_gt, gt in enumerate(self.gt_poly):
            if idx_gt not in self.dt_and_gt_assignation:
                self.list_mse.append(0)
                continue
            x, y = gt.exterior.coords.xy
            # error = self.mapa_color[np.array(y).astype(int),np.array(x).astype(int)]
            error = self.mapa_color[idx_gt]
            mask = ~np.isnan(error)
            error = error[mask]
            mse = (error ** 2).mean()
            self.list_mse.append(np.sqrt(mse))

        plt.figure()
        plt.bar(np.arange(0, len(self.list_mse)), self.list_mse)
        plt.title(f"RMSE global={self.compute_rmse_global():.3f}")
        plt.xlabel('Numero Ring')
        plt.ylabel('RMSE (por gt)')
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/mse.png")
        # plt.show()
        plt.close()

    def ptos_con_angulo_alpha(self, alpha, pts):
        # x,y = pts[:,0], pts[:,1]
        # mask = np.where(self.rayos_img[y,x]==alpha)[0]
        # if len(mask) >= 3:
        #     plt.figure()
        #     plt.imshow(self.im_pre)
        #     plt.scatter(x,y)
        #     plt.figure()
        #     plt.plot(self.rayos_img[y,x])
        #     #plt.show()
        # assert len(mask)<3
        return pts[int(alpha)].reshape(1, 2)

    def polinomio_intermedio(self, pol1, pol2, Nr):
        """pol1 incluido en pol2"""
        pts1 = self.extraer_coordenadas_poligonos(pol1)
        pts2 = self.extraer_coordenadas_poligonos(pol2)

        puntos_nuevo_poligono = []
        for alpha in np.arange(0, 360, 360 / Nr):
            # 1.0 busco todos los puntos que tienen el mismo angulo, basado en la matriz de angulos

            pts1_angulo = self.ptos_con_angulo_alpha(alpha, pts1)
            if len(pts1_angulo) == 0:
                continue
            pts1_media = pts1_angulo.mean(axis=0)
            pts2_angulo = self.ptos_con_angulo_alpha(alpha, pts2)
            if len(pts2_angulo) == 0:
                continue
            pts2_media = pts2_angulo.mean(axis=0)
            # 2.0 busco punto intermedio (media)
            pt_medio = (0.5 * (pts2_media + pts1_media)).astype(int)
            # 3.0 agrego pto medio a poligono list
            puntos_nuevo_poligono.append(pt_medio)
        polygon = Polygon(puntos_nuevo_poligono)
        return polygon

    def _build_influence_area(self, img, gt_poly):
        matriz_influencia = np.zeros((img.shape[1], img.shape[0])) - 1
        gt_poly.sort(key=lambda x: x.area)

        # inicializacion para deliminar regiones
        # from matplotlib.path import Path
        M, N, _ = self.img.shape
        i = 0
        for gt_i in gt_poly:
            gt_i_plus_1 = gt_poly[i + 1] if i < len(gt_poly) - 1 else None
            gt_i_minus_1 = gt_poly[i - 1] if i > 0 else None

            Cm = self.polinomio_intermedio(gt_i, gt_i_minus_1, self.Nr) if gt_i_minus_1 is not None else None
            CM = self.polinomio_intermedio(gt_i, gt_i_plus_1, self.Nr) if gt_i_plus_1 is not None else None

            # self._plot_gt_and_dt_polys(self.im_pre,Cm,CM)
            # TODO  a todos los pixeles delimitados por los poligonos Cm y CM les asigno el valor i
            if CM is None:
                puntos_nuevo_poligono = []
                pts_cm = self.extraer_coordenadas_poligonos(Cm)
                pts_i = self.extraer_coordenadas_poligonos(gt_i)
                for alpha in np.arange(0, 360, 360 / self.Nr):
                    ptsi_angulo = self.ptos_con_angulo_alpha(alpha, pts_i)
                    if len(ptsi_angulo) == 0:
                        continue
                    pts_i_media = ptsi_angulo.mean(axis=0)
                    pts_cm_angulo = self.ptos_con_angulo_alpha(alpha, pts_cm)
                    if len(pts_cm_angulo) == 0:
                        continue
                    pts_cm_media = pts_cm_angulo.mean(axis=0)
                    # 2.0 busco punto intermedio (media)
                    pt_medio = (pts_i_media + (pts_i_media - pts_cm_media)).astype(int)
                    # 3.0 agrego pto medio a poligono list
                    puntos_nuevo_poligono.append(pt_medio)
                CM = Polygon(puntos_nuevo_poligono)

            if Cm is None:
                contours = self.extraer_coordenadas_poligonos(CM)
                mask = np.zeros((N, M))
                cv.fillPoly(mask, pts=[contours], color=(255))


            else:
                contours = self.extraer_coordenadas_poligonos(CM)
                mask_M = np.zeros((N, M))
                cv.fillPoly(mask_M, pts=[contours], color=(255))

                contours = self.extraer_coordenadas_poligonos(Cm)
                mask_m = np.zeros((N, M))
                cv.fillPoly(mask_m, pts=[contours], color=(255))
                mask = mask_M - mask_m

            # plt.figure();plt.imshow(mask);plt.show();plt.close();
            matriz_influencia[mask > 0] = i
            i += 1
        # cv.imwrite("./debug.png", matriz_influencia)
        return matriz_influencia

    def true_positive(self):
        return np.where(np.array(self.dt_and_gt_assignation) > -1)[0].shape[0]

    def false_negative(self):
        gt_ids = np.arange(0, len(self.gt_poly))
        # intersect = np.intersect1d(gt_ids,self.dt_and_gt_assignation)
        dt_asigned = self.true_positive()
        return len(self.gt_poly) - dt_asigned

    def false_positive(self):
        return np.where(np.array(self.dt_and_gt_assignation) == FP_ID)[0].shape[0]

    def true_negative(self):
        return 0

    def precision(self, TP, FP, TN, FN):
        return TP / (TP + FP) if TP + FP > 0 else 0

    def recall(self, TP, FP, TN, FN):
        return TP / (TP + FN) if TP + FN > 0 else 0

    def fscore(self, TP, FP, TN, FN):
        P = self.precision(TP, FP, TN, FN)
        R = self.recall(TP, FP, TN, FN)
        return 2 * P * R / (P + R) if P + R > 0 else 0

    def _asign_gt_to_dt(self, influence_matrix, dt_poly):
        threshold = self.threshold
        self.dt_and_gt_assignation = []
        self.accuracy_percentage = []
        dt_poly.sort(key=lambda x: x.area)
        for poly in dt_poly:
            y, x = poly.exterior.coords.xy
            x = np.array(x).astype(int)
            y = np.array(y).astype(int)
            error_between_dt_and_gt = []
            for gt_poly in self.gt_poly:
                rmse_current = self.compute_rmse_between_dt_and_gt(poly, gt_poly)
                error_between_dt_and_gt.append(rmse_current)

            gts = influence_matrix[x, y].astype(int)
            mask_no_background = np.where(gts >= 0)[0]
            counts = np.bincount(gts[mask_no_background])

            if len(counts) == 0:
                self.dt_and_gt_assignation.append(FP_ID)
                continue

            # gt_label = counts.argmax()
            gt_label = np.argmin(error_between_dt_and_gt)
            if gt_label in self.dt_and_gt_assignation:
                rmse_current = self.compute_rmse_between_dt_and_gt(poly, self.gt_poly[gt_label])
                id_dt_former = np.where(self.dt_and_gt_assignation == gt_label)[0][0]
                rmse_former = self.compute_rmse_between_dt_and_gt(self.dt_poly[id_dt_former], self.gt_poly[gt_label])
                if rmse_current < rmse_former:
                    self.dt_and_gt_assignation[id_dt_former] = FP_ID
                    try:
                        self.accuracy_percentage[id_dt_former] = FP_ID
                    except Exception as e:
                        continue
                else:
                    self.dt_and_gt_assignation.append(FP_ID)
                    self.accuracy_percentage.append(FP_ID)
                    continue

            self.dt_and_gt_assignation.append(gt_label)
            self.accuracy_percentage.append(counts[gt_label] / counts.sum())

        no_asigned_idx = np.where(np.array(self.accuracy_percentage) < threshold)[0].astype(int)
        self.dt_and_gt_assignation = np.array(self.dt_and_gt_assignation)
        self.dt_and_gt_assignation[no_asigned_idx] = FP_ID
        self.accuracy_percentage = np.array(self.accuracy_percentage)
        self.accuracy_percentage[no_asigned_idx] = FP_ID

        self.dt_and_gt_assignation = self.dt_and_gt_assignation.tolist()
        self.accuracy_percentage = self.accuracy_percentage.tolist()
        return 0

    def _asignar_ground_truth_a_detecciones_old(self, matriz_influencias, dt_poly):
        self.dt_and_gt_assignation = []
        self.accuracy_percentage = []
        dt_poly.sort(key=lambda x: x.area)
        for poly in dt_poly:
            y, x = poly.exterior.coords.xy
            x = np.array(x).astype(int)
            y = np.array(y).astype(int)
            gts = matriz_influencias[y, x].astype(int)
            mask_no_background = np.where(gts >= 0)[0]
            counts = np.bincount(gts[mask_no_background])

            if len(counts) == 0:
                self.dt_and_gt_assignation.append(FP_ID)
                continue

            gt_label = counts.argmax()
            if gt_label in self.dt_and_gt_assignation:
                rmse_current = self.compute_rmse_between_dt_and_gt(poly, self.gt_poly[gt_label])
                id_dt_former = np.where(self.dt_and_gt_assignation == gt_label)[0][0]
                rmse_former = self.compute_rmse_between_dt_and_gt(self.dt_poly[id_dt_former], self.gt_poly[gt_label])
                if rmse_current < rmse_former:
                    self.dt_and_gt_assignation[id_dt_former] = FP_ID
                    try:
                        self.accuracy_percentage.pop(id_dt_former)
                    except Exception as e:
                        continue
                else:
                    self.dt_and_gt_assignation.append(FP_ID)
                    continue

            self.dt_and_gt_assignation.append(gt_label)
            self.accuracy_percentage.append(counts[gt_label] / counts.sum())
            x_gt, y_gt = self.gt_poly[gt_label].exterior.coords.xy
            # print(f"dt_count = {len(y)}. gt_count={len(y_gt)}")

    def _plot_assignation_between_gt_and_dt(self):
        import itertools
        import matplotlib.cm as cm
        M, N, _ = self.img.shape
        plt.figure(figsize=(10, 10))
        plt.imshow(np.zeros((M, N)), cmap='gray')

        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        for idx, dt in enumerate(self.dt_poly):
            c = next(colors)
            y, x = dt.exterior.coords.xy
            plt.plot(x, y, color=c)
            # plt.scatter(x,y,s=1)
            if len(self.dt_and_gt_assignation) - 1 < idx:
                continue
            gt_idx = self.dt_and_gt_assignation[idx]
            if gt_idx == FP_ID:
                plt.plot(x, y, color='w')
                continue
            y, x = self.gt_poly[gt_idx].exterior.coords.xy
            plt.plot(x, y, color=c)
            # plt.scatter( x,y, s=1)

        plt.axis('off')
        plt.savefig(f'{self.output_dir}/asignacion_dt_gt.png')
        plt.close()

    def compute_detection_indicators(self):
        # print(f"Porcentaje de Acierto Por Deteccion: {self.accuracy_percentage}")
        nothing = True

    def compute_indicators(self, debug=False):
        influence_matrix = self._build_influence_area(self.img, self.gt_poly)
        self._plot_influecen_area(influence_matrix, self.gt_poly)
        self._asign_gt_to_dt(influence_matrix, self.dt_poly)
        self._plot_assignation_between_gt_and_dt()
        self.compute_detection_indicators()
        TP = self.true_positive()
        FP = self.false_positive()
        TN = self.true_negative()
        FN = self.false_negative()
        self._plot_gt_and_dt_polys(self.img, self.gt_poly, self.dt_poly, n=3, title=f"TP={TP} FP={FP} TN={TN} FN={FN}")
        return TP, FP, TN, FN

    def _plot_influecen_area(self, matriz, list_gt_poly):
        import itertools
        import matplotlib.cm as cm
        M, N, _ = self.img.shape
        plt.figure(figsize=(15, 15))
        plt.imshow(np.zeros((M, N)), cmap='gray')
        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        regiones_id = list(np.unique(matriz).astype(int))
        # regiones_id.remove(-1)

        for region in regiones_id:

            x, y = np.where(matriz == region)
            if region == -1:
                plt.scatter(x, y, s=1, c='w')
            else:
                plt.scatter(x, y, s=1, color=next(colors))

        for poly in list_gt_poly:
            y, x = poly.exterior.coords.xy
            plt.plot(x, y, 'k')

        plt.axis('off')
        plt.savefig(f'{self.output_dir}/areas_influencia.png')
        plt.close()


def extraerPixelesPertenecientesAlPerfil(copia, angle, centro=None):
    """
        angulo =  {0,pi/4,pi/2,3pi/4,pi,5pi/4,6pi/4,7pi/4}
        ptosCard= {S, SE , E  , NE  , width, NW  , W   , SW   }
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


class MetricsDataset:
    def __init__(self, root_dir, creador, config_path, output_dir):
        data_path = get_path("data")
        dataset = pd.read_csv(f"{data_path}/dataset.csv")
        self.creator = creador
        self.results_path = get_path('results')
        self.gt_dir = data_path / f"ground_truth/{creador}/labelme"
        self.data = dataset
        self.root_dir = root_dir
        self.table = pd.DataFrame(columns=['imagen', 'TP', 'FP', 'TN', 'FN', 'P', 'R', 'F'])
        config = load_json(config_path)
        self.Nr = config.get("nr")
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

    def compute(self):
        for index in tqdm(range(self.data.shape[0]), desc='Computing metrics'):
            disco = self.data.iloc[index]
            imagenNombre = disco["Imagen"]
            dt_dir = Path(f"{self.root_dir}/{imagenNombre[:-4]}")
            dt_file = dt_dir / "labelme.json"
            if (not dt_file.exists()):
                continue
            gt_file = self.gt_dir / f"{imagenNombre[:-4]}.json"

            if (not gt_file.exists()):
                row = {'imagen': imagenNombre[:-4], 'TP': None, 'FP': None, 'TN': None, 'FN': None, 'P': None,
                       'R': None, 'F': None}

            else:
                # metrics = ActiveCoutours(gt_file, dt_file)
                img_dir = Path(self.output_dir) / f"{imagenNombre[:-4]}"
                (img_dir).mkdir(exist_ok=True)
                metrics = InfluenceArea(gt_file, dt_file, str(img_dir), self.Nr)
                TP, FP, TN, FN = metrics.compute_indicators()

                # 3.0 compute metrics
                F = metrics.fscore(TP, FP, TN, FN)
                P = metrics.precision(TP, FP, TN, FN)
                R = metrics.recall(TP, FP, TN, FN)
                metrics.compute_error_with_gt()
                metrics.compute_mse_per_gt()
                RMSE = metrics.compute_rmse_global()
                row = {'imagen': imagenNombre[:-4], 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'P': P, 'R': R, 'F': F,
                       'RMSE': RMSE}

                self.table = self.table.append(row, ignore_index=True)

        # Add average
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


def main(rood_dir, creador, config_path, output):
    metrics = MetricsDataset(rood_dir, creador, config_path, output)
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
    dataset = 'X-Pine'
    dataset = 'active_contours'
    root_dir = f'/data/maestria/datasets/cross-section/{dataset}'
    gt_proccessed_path = f'{root_dir}/annotations/processed'
    dataset = pd.read_csv(f"{root_dir}/dataset_ipol.csv")
    data = np.zeros((dataset.shape[0], 6))
    etiquetadores = {}
    images_name = []
    for idx, row in tqdm(dataset.iterrows()):
        disk_name = row.Imagen
        output_filename = f'{root_dir}/annotations/mean_gt/{disk_name}.json'
        # if Path(output_filename).exists():
        #     continue
        # if 'fx'  in disk_name:
        #     continue

        # if disk_name not in ['L07e']:
        #     continue

        creator_list = []
        for creator in ['maria', 'veronica', 'serrana', 'christine']:
            file_gt = f'{gt_proccessed_path}/{creator}/{disk_name}.json'
            if Path(file_gt).exists():
                creator_list.append(file_gt)
            else:
                creator_list.append(None)

        ################################################################################################################
        image_name = f'{root_dir}/images/segmented/{disk_name}.png'
        img_array = cv.imread(image_name)
        height, width, _ = img_array.shape
        ################################################################################################################

        gt_creators = []
        Nr = 360
        centro = [row.cx, row.cy]
        rays_list = build_rays(Nr, width, height, centro)
        draw_ray_curve_and_intersections([], rays_list, [], img_array, "./debug_rays.png")
        # MeanDisk.draw_ray_curve_and_intersections(rays_list, creator_gt,
        #                                           [], [], img_array, './debug.png')
        for idx_creator, creator_file in enumerate(creator_list):
            if creator_file is None:
                continue
            helper = InfluenceArea(creator_file, '', '', Nr, disk_name)
            creator_points = helper.load_ring_stimation(creator_file)
            creator_gt = [helper._sampling_poly(poly, row.cy, row.cx, rays_list, img_array) for poly in creator_points]
            MeanDisk.draw_ray_curve_and_intersections(rays_list, creator_gt,
                                                       [], [], img_array, './debug.png')
            data[idx, idx_creator] = len(creator_gt)
            gt_creators.append(creator_gt)

        ################################################################################################################
        # computar gt medio
        print(output_filename)
        mean_disk = MeanDisk(gt_creators, height, width, row.cx, row.cy, image_name, output_filename, img_array)
        data[idx, [4, 5]] = [height, width]
        images_name.append(disk_name)

    # create a Pandas DataFrame from the NumPy array
    df = pd.DataFrame(data, columns=['maria', 'veronica', 'serrana', 'christine','img_height', 'width'])
    df['imagen'] = images_name
    df.set_index('imagen', inplace=True)
    #df.to_csv(f'{root_dir}/annotations/mean_gt/summary.txt')
    #np.savetxt(f'{dt_dir}/annotations/mean_gt/summary.txt', data)



def process_gt():
    gt_proccessed_path = '/data/maestria/database_new_nomenclature/ground_truth_processed_pith'
    dataset = pd.read_csv(f"/data/maestria/database_new_nomenclature/dataset_ipol.csv")
    data = np.zeros((dataset.shape[0], 6))
    etiquetadores = {}
    for idx, row in tqdm(dataset.iterrows()):
        disk_name = row.Imagen
        # if 'fx' not in disk_name:
        #     continue

        # if disk_name not in ['1-s2.0-S0168169915002847-fx2_lrg']:
        #     continue

        creator_list = []
        for creator in ['mean']:
            file_gt = f'{gt_proccessed_path}/{disk_name}_{creator}.json'
            if Path(file_gt).exists():
                creator_list.append(file_gt)
            else:
                creator_list.append(None)

        ################################################################################################################
        image_name = f'/data/maestria/database_new_nomenclature/images/{disk_name}.png'
        #print(image_name)
        img_array = cv.imread(image_name)
        height, width, _ = img_array.shape
        ################################################################################################################

        gt_creators = []
        Nr = 360
        centro = [row.cx, row.cy]
        rays_list = build_rays(Nr, width, height, centro)
        draw_ray_curve_and_intersections([], rays_list, [], img_array, "./debug_rays.png")

        for idx_creator, creator_file in enumerate(creator_list):
            if creator_file is None:
                continue
            helper = InfluenceArea(creator_file, '', '', Nr, disk_name)
            creator_points = helper.load_ring_stimation(creator_file)
            creator_gt = [helper._sampling_poly(poly, row.cy, row.cx, rays_list, img_array) for poly in creator_points]
            MeanDisk.draw_ray_curve_and_intersections(rays_list, creator_gt,
                                                      [], [], img_array, './debug.png')
            data[idx, idx_creator] = len(creator_gt)
            gt_creators.append(creator_gt)

        ################################################################################################################
        # computar gt medio
        output_filename = f'{gt_proccessed_path}/{disk_name}_mean.json'
        print(output_filename)
        mean_disk = MeanDisk(gt_creators, height, width, row.cx, row.cy, image_name, output_filename, img_array)
        #data[idx, [4, 5]] = [height_output, width_output]

    #np.savetxt(f'{gt_proccessed_path}/comparison/data.txt', data)


class MeanDisk:
    def __init__(self, annotations_lists, height, width, cy, cx, image_path, output_filename, img, nr=360):
        # change grouping of annotations. From creators gt to ring gt.
        self.img = img
        disk_rings_list = self.grouping_gt_annotations(annotations_lists)

        # compute mean ring
        mean_ring_gt = self.compute_mean_gt(nr, height, width, cy, cx, disk_rings_list)
        ############################################################################################################
        # save gt to labelme
        self.save_gt_to_labelme(mean_ring_gt, image_path, height, width, output_filename=output_filename)
        self.mean_ring_gt = mean_ring_gt

    @staticmethod
    def grouping_gt_annotations(annotations_lists):
        disk_rings_list = []
        total_ring_per_disk = len(annotations_lists[0])
        # sort disk
        for idx in range(len(annotations_lists)):
            annotations_lists[idx].sort(key=lambda x: x.area)

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
            for direction in np.arange(0, nr):
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
        # self.draw_ray_curve_and_intersections( rays_list, mean_ring_gt,poly_list,self.intersection_full_list ,self.im_pre, './debug_1.png')

        return mean_ring_gt

    @staticmethod
    def save_gt_to_labelme(mean_ring_gt, image_path, height, width, output_filename):
        labelme_json = {"imagePath": image_path, "imageHeight": height,
                        "imageWidth": width, "version": "5.0.1",
                        "flags": {}, "shapes": [], "imageData": None}

        for idx, mean_ring in enumerate(mean_ring_gt):
            anillo = {"label": str(idx + 1)}
            y, x = mean_ring.exterior.coords.xy
            anillo["points"] = [[j, i] for i, j in zip(y, x)]
            anillo["shape_type"] = "polygon"
            anillo["flags"] = {}
            labelme_json["shapes"].append(anillo)

        write_json(labelme_json, output_filename)

        return

    @staticmethod
    def compute_intersections(gt_ring, rays_list, cy, cx, img=None):
        center = [cy, cx][::-1]
        if not gt_ring.contains(Point(center)):
            return
        # draw_ray_curve_and_intersections([], rays_list, [gt_ring.exterior.coords], im_pre, "./debug_rays.png")
        chain_id = -1
        intersection_list = []
        # draw_ray_curve_and_intersections([], rays_list, [gt_ring.exterior.coords], im_pre, "./debug_rays.png")
        for radii in rays_list:
            try:
                inter = radii.intersection(gt_ring)

            except Exception:
                draw_ray_curve_and_intersections([], [radii], [gt_ring.exterior.coords], img, "./debug_rays.png")
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
                draw_ray_curve_and_intersections([], [radii], [gt_ring.exterior.coords], img, "./debug_rays.png")
                return None

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

        if 'fx' not in disk_name:
            continue

        if disk_name not in ['1-s2.0-S0168169915002847-fx6_lrg']:
            continue

        print(disk_name)

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

        img_array = cv.imread(image_name, cv.COLOR_RGB2GRAY)
        if img_array is None:
            break
        img = np.zeros_like(img_array)
        height, width, _ = img_array.shape
        img[:, :, 0] = img_array[:, :, 0]
        img[:, :, 1] = img_array[:, :, 1]
        img[:, :, 2] = img_array[:, :, 2]
        centro = [row.cx, row.cy]
        Nr = 360
        rays_list = build_rays(Nr, width, height, centro)
        draw_ray_curve_and_intersections([], rays_list, [], img, "./debug_rays.png")
        color = Color()
        for creator in creator_list:
            helper = InfluenceArea(creator, '', '', Nr, disk_name)
            creator_points = helper.load_ring_stimation(creator)
            creator_gt = [helper._sampling_poly(poly, row.cy, row.cx, rays_list, img_array) for poly in creator_points]
            print(len(creator_gt))
            creator_color = color.get_next_color() if 'mean' not in Path(creator).name else Color.black
            for gt in creator_gt:
                img = Drawing.curve(gt.exterior.coords, img_array, creator_color, thickness=1)

        output_filename = f'{str(gt_processed_path_images)}/{disk_name}_creators_only.png'
        cv.imwrite(output_filename, img)

        pdf.add_page()
        pdf.cell(0, 0, disk_name)
        pdf.image(output_filename, x, y, h=height_pdf)

    pdf.output(f"{str(gt_processed_root_path)}/postprocessed_gt.pdf", 'F')

    return


def process_gt_files():
    creators = ['maria', 'veronica', 'christine', 'serrana']
    data_path = get_path("data")
    dataset = pd.read_csv(f"{data_path}/dataset.csv")
    gts_path = '/data/maestria/datasets/ground_truth/analisis'

    gt_maria = f"{gts_path}/maria/results_maria.csv"
    df_maria = pd.read_csv(gt_maria)

    gt_veronica = f"{gts_path}/veronica/results_veronica.csv"
    df_veronica = pd.read_csv(gt_veronica)

    gt_serrana = f"{gts_path}/serrana/results_serrana.csv"
    df_serrana = pd.read_csv(gt_serrana)

    gt_christine = f"{gts_path}/christine/results_christine.csv"
    df_christine = pd.read_csv(gt_christine)

    table = pd.DataFrame(columns=['imagen', 'maria', 'veronica', 'serrana', 'christine'])
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

        data = {'imagen': disk_name, 'christine': christine, 'maria': maria, 'serrana': serrana, 'veronica': veronica}
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
            os.system(
                f"cp {gt_disk_file_name} /data/maestria/datasets/ground_truth/analisis/gt_json/{disk_name}_{creator}.json")


def main(rood_dir, creador, config_path, output):
    metrics = MetricsDataset(rood_dir, creador, config_path, output)
    metrics.compute()
    metrics.print_results()
    metrics.save_results()

    return


class MetricsDataset_over_detection_files:
    def __init__(self, disk_name, detection_files_list, filename_path):
        root_dir = "/data/maestria/database_new_nomenclature/ground_truth_processed/annotations"
        self.file_name_path = filename_path
        config_path = "./config/general.json"
        self.disk_name = disk_name
        self.results_path = get_path('results')
        self.gt_dir = root_dir
        self.root_dir = root_dir
        self.table = pd.DataFrame(columns=['imagen', 'TP', 'FP', 'TN', 'FN', 'P', 'R', 'F'])
        config = load_json(config_path)
        self.Nr = config.get("nr")
        self.detection_files_list = detection_files_list

    def compute(self):
        for dt_file in tqdm(self.detection_files_list, desc='Computing metrics'):
            gt_file = Path(self.gt_dir) / f"{self.disk_name}_mean.json"
            if (not gt_file.exists()) or (not dt_file.exists()):
                row = {'imagen': self.disk_name, 'TP': None, 'FP': None, 'TN': None, 'FN': None, 'P': None, 'R': None,
                       'F': None}

            else:
                img_dir = dt_file.parent
                (img_dir).mkdir(exist_ok=True)
                metrics = InfluenceArea(gt_file, dt_file, str(img_dir), self.Nr)
                TP, FP, TN, FN = metrics.compute_indicators()

                # 3.0 compute metrics
                F = metrics.fscore(TP, FP, TN, FN)
                P = metrics.precision(TP, FP, TN, FN)
                R = metrics.recall(TP, FP, TN, FN)
                metrics.compute_error_with_gt()
                metrics.compute_mse_per_gt()
                RMSE = metrics.compute_rmse_global()

                row = {'imagen': dt_file.parent.name, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'P': P, 'R': R, 'F': F,
                       'RMSE': RMSE, 'exec_time': load_json(str(dt_file))['exec_time(s)']}

            self.table = self.table.append(row, ignore_index=True)

        # Add average
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
        self.table.to_csv(self.file_name_path, sep=',', float_format='%.3f')


class MetricsDataset_comparison:
    def __init__(self, dt_dir, gt_root_dir, config_path, output_dir, threshold=0.75):
        dataset = pd.read_csv(f"{gt_root_dir}/dataset_ipol.csv")
        self.gt_dir = f"{gt_root_dir}/annotations/mean_gt"
        self.data = dataset
        self.dt_dir = dt_dir
        self.table = pd.DataFrame(columns=['imagen', 'TP', 'FP', 'TN', 'FN', 'P', 'R', 'F'])
        config = load_json(config_path)
        self.Nr = config.get("nr")
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        self.threshold = threshold

    def compute(self):
        for index in tqdm(range(self.data.shape[0]), desc='Computing metrics'):
            disco = self.data.iloc[index]
            disk_name = disco["Imagen"]
            # if 'fx' in disk_name:
            #     continue
            # if 'L07e' not in disk_name:
            #     continue
            dt_file = Path(f"{self.dt_dir}/{disk_name}/labelme.json")
            gt_file = Path(self.gt_dir) / f"{disk_name}.json"
            if (not gt_file.exists()) or (not dt_file.exists()):
                row = {'imagen': disk_name, 'TP': None, 'FP': None, 'TN': None, 'FN': None, 'P': None, 'R': None,
                       'F': None}

            else:
                img_dir = Path(self.output_dir) / f"{disk_name}"
                (img_dir).mkdir(exist_ok=True)
                metrics = InfluenceArea(gt_file, dt_file, str(img_dir), self.Nr, self.threshold)
                TP, FP, TN, FN = metrics.compute_indicators()

                # 3.0 compute metrics
                F = metrics.fscore(TP, FP, TN, FN)
                P = metrics.precision(TP, FP, TN, FN)
                R = metrics.recall(TP, FP, TN, FN)
                metrics.compute_error_with_gt()
                metrics.compute_mse_per_gt()
                RMSE = metrics.compute_rmse_global()
                row = {'imagen': disk_name, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'P': P, 'R': R, 'F': F,
                       'RMSE': RMSE, 'exec_time': load_json(str(dt_file))['exec_time(s)']}

            self.table = self.table.append(row, ignore_index=True)

        # Add average
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
        self.table.to_csv(f"{self.output_dir}/results_th_{self.threshold}.csv",
                          sep=',', float_format='%.3f')


def main_comparison(dt_dir, gt_root_dir, output, threshold=0.60):
    # dt_dir = "/data/maestria/database_new_nomenclature/ground_truth_processed/annotations"
    # dt_dir = f"/data/maestria/resultados/{creator}"
    # gt_root_dir = "/data/maestria/database_new_nomenclature/ground_truth_processed_pith"
    # gt_root_dir = "/data/maestria/datasets/cross-section/X-Pine/"
    # gt_root_dir = "/data/maestria/datasets/cross-section/active_contours/"
    # output = f"/data/maestria/database_new_nomenclature/ground_truth_processed/annotations_new_gt/comparison/{creator}"
    config_path = "./config/general.json"
    metrics = MetricsDataset_comparison(dt_dir, gt_root_dir, config_path, output, threshold)
    metrics.compute()
    metrics.print_results()
    metrics.save_results()

    return


def metric_dependance_with_threshold():
    creator = 'ipol_refactoring_resize_experiment_1500_3'
    th_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for th in th_range:
        print(th)
        main_comparison(creator, th / 100)

    return 0


def mean_gt_generation_and_comparison_between_creator(dt_dir, gt_root_dir, output, th):
    #process_gt()
    #compute_mean_gt()
    # draw_creator_annotation_over_disk()
    main_comparison(dt_dir, gt_root_dir, output, th/100)
    #main_comparison(creator)
    # main_comparison('serrana')
    # main_comparison('christine')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--creator", type=str, required=True)
    #parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dt_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--th", type=int, required=True)

    args = parser.parse_args()
    mean_gt_generation_and_comparison_between_creator(args.dt_dir, args.gt_dir, args.output, args.th)
    #mean_gt_generation_and_comparison_between_creator('')
    # main(str(args.dt_dir),'mean',str(args.config_path), str(args.output))
    # metric_dependance_with_threshold()
    # process_gt_files()
    # cp_gt_files_to_new_location()
