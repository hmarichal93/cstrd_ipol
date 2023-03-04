import numpy as np
import cv2
from shapely.geometry import Polygon, Point
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


from lib.sampling import build_rays, draw_ray_curve_and_intersections
from lib.io import get_path, load_json
import lib.chain as ch
from lib.drawing import Drawing, Color
from lib.metrics_new_nomenclature import InfluenceArea, MetricsDataset_over_detection_files


def processing_tables():
    img_res_dir = Path("/data/maestria/database_new_nomenclature/ground_truth_processed/"
                       "annotations/comparison/ipol_refactoring_resize_experiment_1500_3")
    img_res_dir.mkdir(exist_ok=True)
    metadata_filename = get_path('metadata') / 'dataset_ipol.csv'
    metadata = pd.read_csv(metadata_filename)
    th_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    #plt.figure()
    image_list = []# ['F02a','F02b','F02c','F02d','F02e','F03c','F07b','L02b','L03c','F03d']
    fscore_matrix = []
    rmse_matrix = []
    for th in th_range:
        filename_th = img_res_dir / f"results_ipol_refactoring_resize_experiment_1500_3_th_{th/100}_contornos_activos.csv"
        data = pd.read_csv(filename_th)
        fscore_matrix.append(data[data.imagen == 'Average'].F)
        rmse_matrix.append(data[data.imagen == 'Average'].RMSE)

    plt.figure()
    plt.plot(th_range,fscore_matrix)
    plt.grid(True)
    plt.ylabel('F-Score')
    plt.xlabel('Threshold')
    plt.show()

    plt.figure()
    plt.plot(th_range, rmse_matrix)
    plt.grid(True)
    plt.ylabel('rmse')
    plt.xlabel('Threshold')
    plt.show()

    return 0

if __name__=="__main__":
    processing_tables()