import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry.linestring import LineString

from lib.drawing import Color

DELIMITE_CURVE_ROW = np.array([-1, -1])


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

def mask_background(img):
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    mask[ img[ :, :] == Color.gray_white ] = Color.gray_white
    return mask

def blur(img, blur_size=11):
    return cv2.GaussianBlur(img, (blur_size, blur_size), 0)

def thresholding(mask, threshold=0):
    mask = np.where(mask>threshold,Color.gray_white,0).astype(np.uint8)
    return mask
def padding_mask(mask):
    pad = 3
    mask = np.pad(mask, ( (pad,pad), (pad,pad) ), mode='constant', constant_values=255).astype(np.uint8)
    return mask
def find_border_contour(mask, img):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    perimeter_image = 2*img.shape[0]+2*img.shape[1]
    error_threshold = 0.1*perimeter_image
    max_cont = None
    area_difference_min = np.inf
    area_image = img.shape[0] * img.shape[1]
    approximate_disk_area = area_image / 2
    for idx, c in enumerate(contours):
        contour_area = cv2.contourArea(c)
        contour_perimeter = cv2.arcLength(c, True)
        if np.abs(contour_perimeter-perimeter_image) < error_threshold:
            # contour is the border of the image. Need to discart it
            continue

        area_difference = np.abs(contour_area-approximate_disk_area)

        if area_difference < area_difference_min:
            max_cont = c.reshape((-1, 2))
            area_difference_min = area_difference

    return max_cont
def contour_to_curve(contour, name):
    curve = Curve(contour, name)
    return curve
def get_border_curve(img, l_ch_f):
    """
    Get disk border border_curve of the image
    @param img: segmented gray image
    @param l_ch_f: list of curves
    @return: border object border_curve

    """
    mask = mask_background(img)
    mask = blur(mask)
    mask = thresholding(mask)
    mask = padding_mask(mask)
    border_contour = find_border_contour(mask, img)
    border_curve = contour_to_curve(border_contour, len(l_ch_f))
    return border_curve


def change_reference_axis(ch_e_matrix, cy, cx):
    center = [cx, cy]
    curve_border_index = np.where(ch_e_matrix == DELIMITE_CURVE_ROW)[0]
    X = ch_e_matrix.copy()

    #change reference axis
    Xb = np.array([[1, 0], [0, 1]]).dot(X.T) + (np.array([-1, -1]) * np.array(center, dtype=float)).reshape(
        (-1, 1))

    #mask delimiting edge row by -1
    Xb[:,curve_border_index] = -1
    return Xb

def convert_masked_pixels_to_curves(X_edges_filtered):
    curve_border_index = np.unique(np.where(X_edges_filtered == DELIMITE_CURVE_ROW)[0])
    start = -1
    ch_f = []
    for end in curve_border_index:
        if end - start > 2:
            pixel_list = X_edges_filtered[start + 1:end].tolist()
            curve = Curve(pixel_list, len(ch_f))
            ch_f.append(curve)
        start = end

    return ch_f
def get_gradient_vector_for_each_edge_pixel(ch_e, Gx, Gy):
    G = np.vstack(
        (Gx[ch_e[:, 1].astype(int), ch_e[:, 0].astype(int)], Gy[ch_e[:, 1].astype(int), ch_e[:, 0].astype(int)])).T
    return G

def compute_angle_beetween_gradient_and_edges(Xb_normed, gradient_normed):
    theta = np.arccos(np.clip((gradient_normed * Xb_normed).sum(axis=1), -1.0, 1.0)) * 180 / np.pi
    return theta
def filter_edges_by_threshold(m_ch_e, theta, alpha):
    X_edges_filtered = m_ch_e.copy()
    X_edges_filtered[theta >= alpha] = -1
    return X_edges_filtered
def filter_edges(m_ch_e, cy, cx, Gx, Gy, alpha, im_pre):
    """
    Edge detector find three types of edges: early wood transitions, latewood transitions and radial edges produced by
    cracks and fungi. Only early wood edges are the ones that forms the rings. In other to filter the other ones
    collineary with the ray direction is computed and filter depending on threshold (alpha)
    @param m_ch_e: devernay curves in matrix format
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param Gx: Gradient over x direction
    @param Gy: Gradient over y direction
    @param alpha: threshold filter
    @param im_pre: input image
    @return:
    - l_ch_f: filtered devernay curves
    """
    #1.0 change reference axis
    Xb = change_reference_axis(m_ch_e, cy, cx)
    #2.0 get normalized gradient at each edge
    G = get_gradient_vector_for_each_edge_pixel(m_ch_e, Gx, Gy)
    #3.0 Normalize gradient and rays
    Xb_normalized = normalized_row_matrix(Xb.T)
    G_normalized = normalized_row_matrix(G)
    #4.0 Compute angle between gradient and edges
    theta = compute_angle_beetween_gradient_and_edges(Xb_normalized, G_normalized)
    #5.0 filter pixels by threshold
    X_edges_filtered = filter_edges_by_threshold(m_ch_e, theta, alpha)
    #5.0 Convert masked pixel to object curve
    l_ch_f = convert_masked_pixels_to_curves(X_edges_filtered)
    #6.0 Border disk is added as a curve
    border_curve = get_border_curve(im_pre, l_ch_f)
    l_ch_f.append(border_curve)

    return l_ch_f

def write_filter_curves_to_image(curves, img):
    img = np.zeros_like(img) +255
    for c in curves:
        img = c.draw(img,thickness=3)
    return img
