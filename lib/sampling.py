import numpy as np
import time
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from lib.drawing import Drawing
import cv2

from lib.chain import Node, euclidean_distance, get_node_from_list_by_angle, Chain, TypeChains, visualize_chains_over_image


class Ray(LineString):
    def __init__(self, direction, center, M, N):
        self.direction = direction
        self.border = self._image_border_radii_intersection(direction, center, M, N)
        super().__init__([center,self.border])


    @staticmethod
    def _image_border_radii_intersection(theta, origin, M, N):
        degree_to_radians = np.pi/180
        theta = theta % 360
        yc,xc = origin
        if 0 <= theta < 45:
            ye = M-1
            xe = np.tan(theta*degree_to_radians) * (M-1-yc) + xc

        elif 45<= theta < 90:
            xe = N-1
            ye = np.tan((90-theta)*degree_to_radians)*(N-1-xc) + yc

        elif 90<= theta < 135:
            xe = N-1
            ye = yc - np.tan((theta-90)*degree_to_radians)*(xe-xc)

        elif 135 <= theta < 180:
            ye = 0
            xe = np.tan((180-theta)*degree_to_radians)*(yc) + xc

        elif 180 <= theta < 225:
            ye =0
            xe = xc- np.tan((theta-180)*degree_to_radians)*(yc)

        elif 225 <= theta < 270:
            xe = 0
            ye = yc - np.tan((270-theta)*degree_to_radians)*(xc)

        elif 270 <= theta < 315:
            xe = 0
            ye = np.tan((theta-270) * degree_to_radians) * (xc) + yc

        elif 315 <= theta < 360:
            ye = M-1
            xe = xc - np.tan((360-theta) * degree_to_radians) * (ye - yc)

        else:
            raise 'Error'

        return (ye,xe)
def build_rays(Nr, M, N, center):
    """

    @param Nr: total rays
    @param N: widht image
    @param M: height image
    @param center: (y,x)
    @return: list_position rays
    """
    angles_range = np.arange(0, 360, 360 / Nr)
    radii_list = [Ray(direction, center, M, N) for direction in angles_range]
    return radii_list

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
def intersections_between_rays_and_devernay_curves(center, radii_list, curve_list, min_chain_lenght, nr, height, witdh):
    chain_list, dot_list = [], []
    for idx,curve in enumerate(curve_list):
        curve_dots = []
        chain_id = len(chain_list)
        for radii in radii_list:
            inter = radii.intersection(curve)
            if not inter.is_empty:
                try:
                    y,x = get_coordinates_from_intersection(inter)
                except NotImplementedError:
                    continue
                i, j = np.array(y)[0],np.array(x)[0]
                params = {'y': i, 'x': j, 'angle': int(radii.direction), 'radial_distance':
                    euclidean_distance([ i, j], center),'chain_id': chain_id}

                dot = Node(**params)
                if dot not in curve_dots and get_node_from_list_by_angle(curve_dots, radii.direction) is None:
                    curve_dots.append(dot)

            # elif idx == len(curve_list) -1:
            #     print(radii.direction)

        if len(curve_dots) < min_chain_lenght:
            continue

        dot_list += curve_dots
        chain = Chain(chain_id, nr, center=center, M=height, N=witdh)
        chain.add_nodes_list(curve_dots)
        chain_list.append(chain)

    chain_list[-1].type = TypeChains.border

    return dot_list, chain_list

def generate_virtual_center_chain(cy, cx, nr , chains_list, dots_list,  height, witdh):
    chain_id = len(chains_list)-1
    center_list = [Node(**{'x': cx, 'y': cy, 'angle': angle, 'radial_distance': 0,
                  'chain_id': chain_id}) for angle in np.arange(0,360, 360/nr)]
    dots_list += center_list

    chain = Chain(chain_id, nr,center=chains_list[0].center, M=height, N=witdh, type=TypeChains.center)
    chain.add_nodes_list(center_list)

    chains_list.append(chain)

    #set border chain as the last element of the list
    chains_list[-2].change_id(len(chains_list) - 1)


    return 0


def draw_ray_curve_and_intersections(dots_lists, rays_list, curves_list, img, filename):
    img_draw = img.copy()
    for ray in rays_list:
        img_draw = Drawing.radii(ray, img_draw)

    for curve in curves_list:
        img_draw = Drawing.curve(curve, img_draw)

    for dot in dots_lists:
        img_draw = Drawing.intersection(dot, img_draw)

    cv2.imwrite(filename, img_draw)
def sampling_edges(ch_f, cy, cx, nr, min_chain_lenght, im_pre, debug=False):
    """
    Devernay curves are sampled using the rays directions.
    @param ch_f:  edges devernay curves
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param nr: total ray number
    @param min_chain_lenght:  minumin chain lenght
    @param im_pre: input image
    @param debug: debugging flag
    @return:
    - ch_s: sampled edges curves. List of chain object
    - nodes_s: nodes list.
    """
    height, witdh = im_pre.shape
    rays_list = build_rays(nr, height, witdh,   [cy, cx])
    nodes_s, ch_s = intersections_between_rays_and_devernay_curves([cy, cx], rays_list, ch_f, min_chain_lenght, nr, height, witdh)
    generate_virtual_center_chain(cy, cx, nr, ch_s, nodes_s, height, witdh)
    if debug:
        draw_ray_curve_and_intersections(nodes_s, rays_list, ch_f, im_pre, './dots_curve_and_rays.png')

    return ch_s, nodes_s
