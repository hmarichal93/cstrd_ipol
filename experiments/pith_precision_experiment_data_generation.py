import numpy as np
import cv2
from shapely.geometry import Polygon, Point
import os
import pandas as pd

from lib.sampling import build_rays
from lib.io import get_path
import lib.chain as ch
from lib.drawing import Drawing, Color
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
def return_list_of_experimental_center( img, cy,cx, filename, total_directions=10, max_error = 50):
    """
    Given ground truth pith position.
    @return:
    """
    #direction_rays = np.arange(0, 360, 360/total_directions)
    height, witdh = img.shape
    rays_list = build_rays(total_directions, height, witdh, [cy, cx])
    border_disk = get_disk_border(img)

    step = max_error / total_directions
    error_percentage_range = np.arange(5,max_error,step)
    center = [cy, cx]
    center_error = []
    for error_percentage in error_percentage_range:
        for radii in rays_list:
            inter = radii.intersection(border_disk)
            y,x = get_coordinates_from_intersection(inter)
            i, j = np.array(y)[1], np.array(x)[1]
            radial_distance = ch.euclidean_distance([i, j], center)
            radial_new_dot = radial_distance * error_percentage / 100
            y, x = from_polar_to_cartesian(radial_new_dot, radii.direction, center)
            center_error.append([y,x])



    ######
    img_debug = np.zeros((height, witdh, 3))
    img_debug[:,:,0] = img
    img_debug[:, :, 1] = img
    img_debug[:, :, 2] = img
    for center_dt in center_error:
        cy_dt, cx_dt = int(center_dt[0]), int(center_dt[1])
        img_debug = Drawing.circle(img_debug, center_coordinates = (cy_dt, cx_dt), color=Color.red, thickness=-1, radius=3 )

    cv2.imwrite(filename, img_debug)


    return center_error

if __name__=="__main__":
    sigma = 3
    res_size = 1500
    img_res_dir = get_path("results") / "pith_exp"
    img_res_dir.mkdir(exist_ok=True)
    metadata_filename = get_path('metadata') / 'dataset_ipol.csv'

    metadata = pd.read_csv(metadata_filename)
    image_list = ['F02a','F02b','F02c','F02d','F02e','F03c','F07b','L02b','L03c','F03d']
    for idx_images in range(metadata.shape[0]):
        row = metadata.iloc[idx_images]
        img_name = row.Imagen
        if img_name not in image_list:
            continue


        ################################################################################################################
        filename = f"/data/maestria/database_new_nomenclature/images/{img_name}.png"
        img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        cy, cx = row.cy, row.cx
        center_error = return_list_of_experimental_center(img_array, cy, cx,str(img_res_dir/f"{img_name}.png"))
        for idx, center_dt in enumerate(center_error):
            cy_dt, cx_dt = int(center_dt[1]), int(center_dt[0])
            exp_dir = img_res_dir / f"{img_name}_{idx}"

            exp_dir.mkdir(exist_ok=True)
            labelme_file = exp_dir / "labelme.json"
            if labelme_file.exists():
                continue
            command = f"python main.py --input {filename} --sigma {sigma} --cy {cy_dt} --cx {cx_dt}  --root ./ --output_dir" \
                      f" {str(exp_dir)} --hsize {res_size} --wsize {res_size} --save_imgs 1"
            print(command)
            os.system(command)