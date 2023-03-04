import numpy as np
import cv2
from shapely.geometry import Polygon, Point
import os
import pandas as pd

from lib.sampling import build_rays, draw_ray_curve_and_intersections
from lib.io import get_path, load_json
import lib.chain as ch
from lib.drawing import Drawing, Color
from lib.metrics_new_nomenclature import InfluenceArea, MetricsDataset_over_detection_files
def get_disk_border(img):
    """
    @param img: segmented gray image
    @param curves_list:
    @return:

    """
    background_color = img[0, 0]
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    mask[ img[ :, :] == background_color] = 255


    blur = 11
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    mask = np.where(mask>0,255,0).astype(np.uint8)
    pad = 3
    mask = np.pad(mask,((pad,pad),(pad,pad)), mode='constant', constant_values=255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    perimeter_image = 2*img.shape[0]+2*img.shape[1]
    # find the biggest contour
    max_cont = None
    area_difference_min = np.inf
    approximate_disk_area = np.pi * (img.shape[0] / 2) ** 2
    area_image = img.shape[0] * img.shape[1]
    approximate_disk_area = area_image / 2
    for idx,c in enumerate(contours):
        contour_area = cv2.contourArea(c)
        contour_perimeter = cv2.arcLength(c, True)
        if np.abs(contour_perimeter-perimeter_image)<0.1*perimeter_image:
            continue

        area_difference = np.abs(contour_area-approximate_disk_area)
        # print(f"{idx} Area {area_difference} {contour_area}")
        # image = img.copy()
        # cv2.drawContours(image, c, -1, (0, 255, 0), 1)
        # cv2.imwrite(f'./output/mask_{idx}.png', image)
        if area_difference < area_difference_min:
            max_cont = c.reshape((-1, 2))
            max_contour_draw = c
            area_difference_min = area_difference

    return Polygon(max_cont)
def from_polar_to_cartesian(r,angulo,centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y,x)

def get_coordinates_from_intersection(inter):
    if 'MULTI' in inter.wkt:
        inter = inter[0]

    if type(inter) == Point:
        y, x = inter.xy

    elif 'LINESTRING' in inter.wkt:
        y, x = inter.xy

    elif 'STRING' in inter.wkt:
        y, x = inter.coords.xy

    else:
        raise

    return y,x
def return_list_of_experimental_center( img, cy,cx, filename, gt_filename, total_directions=6, max_error = 50):
    """
    Given ground truth pith position.
    @return:
    """

    # 1.0 load gt and select only first and second ring
    gt_polis = InfluenceArea.load_ring_stimation(gt_filename)
    gt_polis.sort(key=lambda x: x.area)

    # 2.0 select ray direction
    height, witdh = img.shape
    rays_list = build_rays(total_directions, height, witdh, [cx, cy])

    ###
    img_debug = np.zeros((height, witdh, 3))
    img_debug[:,:,0] = img
    img_debug[:, :, 1] = img
    img_debug[:, :, 2] = img


    ###
    rings_nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    center = [cy, cx]
    # 3.0
    dictionary_centers = {}
    inner_idx_ring = 1
    while True:
        idx_gt, poly = (0, gt_polis[0]) if inner_idx_ring in [1,2,3,4] else (1, gt_polis[1])
        if inner_idx_ring>8:
            break

        # 3.1 compute center over nodes rings
        center_error = []
        for radii in rays_list:
            inter = radii.intersection(poly)
            y, x = get_coordinates_from_intersection(inter)
            j, i = np.array(y)[1], np.array(x)[1]


            if idx_gt == 0:
                radial_distance = ch.euclidean_distance([i, j], center)
                radial_semiline = radial_distance / 4
                if inner_idx_ring in [1, 2, 3]:
                    x_dt, y_dt = from_polar_to_cartesian( inner_idx_ring*radial_semiline, radii.direction, center[::-1])
                elif inner_idx_ring == 4:
                    y_dt, x_dt = i, j
            else:
                poly_1 = gt_polis[0]
                inter = radii.intersection(poly_1)
                y, x = get_coordinates_from_intersection(inter)
                j_gt, i_gt = np.array(y)[1], np.array(x)[1]
                radial_distance = ch.euclidean_distance([i_gt, j_gt], center)
                radial_distance_2 = ch.euclidean_distance([i, j], center)
                radial_semiline = (radial_distance_2 - radial_distance) / 4
                if inner_idx_ring in [5, 6, 7]:
                    if inner_idx_ring == 5:
                        inner_idx_ring_second_ring = 1

                    elif inner_idx_ring == 6:
                        inner_idx_ring_second_ring = 2

                    elif inner_idx_ring == 7:
                        inner_idx_ring_second_ring = 3

                    x_dt, y_dt = from_polar_to_cartesian( inner_idx_ring_second_ring*radial_semiline + radial_distance, radii.direction, center[::-1])

                else:
                    y_dt, x_dt = i, j

            center_error.append([ y_dt, x_dt])

        dictionary_centers[inner_idx_ring] = center_error
        inner_idx_ring += 1


    ######


    for key in dictionary_centers.keys():
        center_error = dictionary_centers[key]
        for center_dt in center_error:
            cy_dt, cx_dt = int(center_dt[0]), int(center_dt[1])
            img_debug = Drawing.circle(img_debug, center_coordinates = (cy_dt, cx_dt), color=Color.red, thickness=-1, radius=3 )
    img_debug = Drawing.circle(img_debug, center_coordinates=(cy, cx), color=Color.red, thickness=-1, radius=3)
    draw_ray_curve_and_intersections([], rays_list,[curve.exterior.coords for curve in gt_polis],img_debug,filename)
    return dictionary_centers

from tqdm import tqdm
from pathlib import Path
def main():
    sigma = 3
    res_size = 1500
    img_res_dir = get_path("results") / "pith_exp"
    img_res_dir.mkdir(exist_ok=True)
    metadata_filename = get_path('metadata') / 'dataset_ipol.csv'
    metadata = pd.read_csv(metadata_filename)
    #image_list = ['F02a','F02b','F02c','F02d','F02e','F03c','F07b','L02b','L03c','F03d']
    for idx_images in tqdm(range(metadata.shape[0])):
        row = metadata.iloc[idx_images]
        img_name = row.Imagen
        # if img_name not in image_list:
        #     continue
        ################################################################################################################
        filename = f"/data/maestria/database_new_nomenclature/images/{img_name}.png"

        img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gt_file = f"/data/maestria/database_new_nomenclature/ground_truth_processed/annotations/{img_name}_mean.json"
        if not Path(gt_file).exists():
            continue

        cy, cx = row.cy, row.cx
        dictionray_center = return_list_of_experimental_center(img_array, cy, cx, str(img_res_dir / f"{img_name}.png"),
                                                               gt_file)
        for key in dictionray_center.keys():
            center_error = dictionray_center[key]
            labelme_files = []
            for idx, center_dt in enumerate(center_error):
                cy_dt, cx_dt = int(center_dt[1]), int(center_dt[0])
                exp_dir = img_res_dir / f"{img_name}_{key}_{idx}"

                exp_dir.mkdir(exist_ok=True)
                labelme_file = exp_dir / "labelme.json"
                if labelme_file.exists():
                    continue
                labelme_files.append(labelme_file)
                command = f"python main.py --input {filename} --sigma {sigma} --cy {cy_dt} --cx {cx_dt}  --root ./ --output_dir" \
                          f" {str(exp_dir)} --hsize {res_size} --wsize {res_size} --save_imgs 1"
                print(command)
                os.system(command)

            if len(labelme_files) == 0:
                continue

            output_filename = img_res_dir / f"{img_name}_error_{int(key)}.csv"
            metrics = MetricsDataset_over_detection_files(img_name, labelme_files, output_filename)
            metrics.compute()
            metrics.print_results()
            metrics.save_results()

    return 0
import matplotlib.pyplot as plt
def processing_tables():
    img_res_dir = get_path("results") / "pith_exp"
    img_res_dir.mkdir(exist_ok=True)
    metadata_filename = get_path('metadata') / 'dataset_ipol.csv'
    disk_with_pith = pd.read_csv(img_res_dir / "disk_with_pith.csv")
    disk_with_pith_list = disk_with_pith.discos.values.tolist()
    metadata = pd.read_csv(metadata_filename)
    error_range = [ 1, 2, 3, 4, 5, 6, 7, 8]
    #plt.figure()
    image_list = ['F02a','F02b','F02c','F02d','F02e','F03c','F07b','L02b','L03c','F03d']
    fscore_matrix = []
    rmse_matrix = []
    for idx_images in range(metadata.shape[0]):
        row = metadata.iloc[idx_images]
        img_name = row.Imagen
        if img_name in disk_with_pith_list:
            continue
        if 'fx' in img_name:
            continue
        # if img_name not in image_list:
        #     continue

        ################################################################################################################
        fscore = []
        rmse = []
        for pos in error_range:
            filename = img_res_dir / f"{img_name}_error_{pos}.csv"
            if not filename.exists():
                print(img_name)
                break
            df = pd.read_csv(filename)
            fscore.append(df[df.imagen == 'Average'].F.values[0])
            rmse.append(df[df.imagen == 'Average'].RMSE.values[0])

        if not filename.exists():
            continue

        fscore_matrix.append(fscore)
        rmse_matrix.append(rmse)
        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################
        #plt.plot( error_range, fscore)

    # plt.grid(True)
    # plt.xlabel('error')
    # plt.ylabel('fscore')
    # plt.show()
    fscore_matrix = np.array(fscore_matrix)
    rmse_matrix = np.array(rmse_matrix)
    fscore_average = fscore_matrix.mean(axis=0)
    fscore_std = fscore_matrix.std(axis=0)
    percentil_cte = 1.97
    fscore_std_list = np.array([0.13] + fscore_std.tolist())
    fscore_average_list = [0.89] + fscore_average.tolist()

    xx = [0] + error_range
    plt.figure()
    plt.plot(xx, fscore_average_list,'r.')
    plt.plot(xx, fscore_average_list,'k', label='mean')
    plt.plot(xx, fscore_std_list*percentil_cte + fscore_average_list,'b',label='std')
    plt.plot(xx, -fscore_std_list * percentil_cte + fscore_average_list, 'b')
    plt.grid(True)
    plt.legend()
    plt.xlabel('error')
    plt.ylabel('Fscore')
    plt.title('Fscore average')
    plt.show()

    rmse_average = rmse_matrix.mean(axis=0)
    rmse_std = rmse_matrix.std(axis=0)
    rmse_std_list = np.array([2.44] + rmse_std.tolist())
    rmse_average_list = [6.22]+rmse_average.tolist()
    plt.figure()
    plt.plot([0]+error_range, rmse_average_list , 'r.')
    plt.plot([0]+error_range,rmse_average_list, 'k',label='mean')
    plt.plot(xx, rmse_std_list * percentil_cte + rmse_average_list, 'b',label='std')
    plt.plot(xx, -rmse_std_list * percentil_cte + rmse_average_list, 'b')
    plt.legend()
    plt.grid(True)
    plt.xlabel('error')
    plt.ylabel('rmse')
    plt.title('RMSE average')
    plt.show()

    return 0

if __name__=="__main__":
    #main()
    processing_tables()