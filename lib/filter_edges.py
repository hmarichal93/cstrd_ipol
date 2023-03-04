import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry.linestring import LineString

from lib.drawing import Color

class Curve(LineString):
    def __init__(self, pixels_list, name):
        self.id = name
        super().__init__(np.array(pixels_list)[:, [1, 0]].tolist())

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)

    def draw(self, img, thickness=1):
        y, x = self.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x,y)).T
        isClosed = False
        img = cv2.polylines(img, [pts],
                              isClosed, Color.black, thickness)
        return img

def normalized_row_matrix(matrix):
    sqrt = np.sqrt((matrix ** 2).sum(axis=1))
    normalized_array = matrix / sqrt[:, np.newaxis]
    return normalized_array


def erosion(erosion_size, src):

    erosion_shape = cv2.MORPH_CROSS

    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erosion_dst = cv2.erode(src, element)
    return erosion_dst


def dilatation(dilatation_size, src):
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(src, element)
    return dilatation_dst

def get_disk_border(img, curves_list):
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
    area_image = img.shape[0] * img.shape[1]
    approximate_disk_area = area_image / 2
    for idx,c in enumerate(contours):
        contour_area = cv2.contourArea(c)
        contour_perimeter = cv2.arcLength(c, True)
        if np.abs(contour_perimeter-perimeter_image)<0.1*perimeter_image:
            continue

        area_difference = np.abs(contour_area-approximate_disk_area)

        if area_difference < area_difference_min:
            max_cont = c.reshape((-1, 2))
            area_difference_min = area_difference

    curve = Curve(max_cont, len(curves_list))
    return curve

def filter_edges(ch_e, cy, cx, Gx, Gy, edges_th, img):
    """
    Edge detector find three types of edges: early wood transitions, latewood transitions and radial edges produced by
    cracks and fungi. Only early wood edges are the ones that forms the rings. In other to filter the other ones
    collineary with the ray direction is computed and filter depending on threshold (edges_th)
    @param ch_e: devernay curves
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param Gx: Gradient over x direction
    @param Gy: Gradient over y direction
    @param edges_th: threshold filter
    @param img: input image
    @return:
    - ch_f: filtered devernay curves
    """
    #1.0 normalize ray vector at each edge
    delimiter_curve_row = np.array([-1, -1])
    center = [cx, cy]
    curve_border_index = np.where(ch_e == delimiter_curve_row)[0]
    X = ch_e.copy()
    X[curve_border_index] = 0
    Xb = np.array([[1, 0], [0, 1]]).dot(X.T) + (np.array([-1, -1]) * np.array(center, dtype=float)).reshape(
        (-1, 1))
    #2.0 get normalized gradient at each edge
    gradient = np.vstack(
        (Gx[X[:, 1].astype(int), X[:, 0].astype(int)], Gy[X[:, 1].astype(int), X[:, 0].astype(int)])).T

    #3.0 Normalize gradient and rays
    Xb_normed = normalized_row_matrix(Xb.T)
    gradient_normed = normalized_row_matrix(gradient)
    #4.0 Compute angle between gradient and edges
    theta = np.arccos(np.clip((gradient_normed * Xb_normed).sum(axis=1), -1.0, 1.0)) * 180 / np.pi

    #5.0 filter pixels by threshold
    X_edges_filtered = ch_e.copy()
    X_edges_filtered[theta > edges_th] = -1

    #5.0 Convert masked pixel to object curve
    curve_border_index = np.unique(np.where(X_edges_filtered == delimiter_curve_row)[0])
    start = -1
    ch_f = []
    for end in curve_border_index:
        if end - start > 2:
            pixel_list = X_edges_filtered[start + 1:end].tolist()
            curve = Curve(pixel_list, len(ch_f))
            ch_f.append(curve)
        start = end
    #6.0 Border disk is added as a curve
    ch_f.append(get_disk_border(img, ch_f))

    return ch_f

def write_filter_curves_to_image(curves, img):
    img = np.zeros_like(img) +255
    for c in curves:
        img = c.draw(img,thickness=3)
    return img
