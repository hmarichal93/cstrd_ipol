import cv2
from PIL import Image
import numpy as np



def resize(im: np.array, newsize, center_y=1, center_x=1):
    hnew, wnew = newsize
    if hnew is None or wnew is None:
        return im, center_y, center_x

    pil_img = Image.fromarray(im)

    if im.ndim > 2:
        height, width, _ = im.shape
    else:
        height, width = im.shape
    #Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    flag = Image.ANTIALIAS
    #flag = Image.Resampling.LANCZOS
    pil_img = pil_img.resize(newsize, flag)
    np_img = np.array(pil_img)

    #center transformation
    hscale = hnew / height
    wscale = wnew / width
    center_y *= hscale
    center_x *= wscale
    return np_img, center_y, center_x
def equalize(imageGray):
    # equalize image
    mask = np.where(imageGray == 255, 1, 0)
    img_eq = imageGray.copy()
    img_eq[mask > 0] = np.mean(img_eq[mask == 0])
    clahe = cv2.createCLAHE(clipLimit=10)
    img_eq = clahe.apply(img_eq)
    img_eq[mask > 0] = 255

    return img_eq

def rgb2gray(img_r):
    return cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
def preprocessing(im_in, height = None, width = None, cy = None, cx = None):
    """
    Image preprocessing steps. Following actions are realized
    - image resize
    - image is converted to gray scale
    - gray scale image is equalized
    @param im_in: segmented image
    @param height: new image height
    @param width: new image widht
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @return:
    - im_eq: equalized image
    - cy: pith y's coordinate after resize
    - cx: pith x's coordinate after resize
    """
    im_r, cy, cx = resize(im_in, (height, width), cy, cx)
    im_g = rgb2gray(im_r)
    im_eq = equalize(im_g)
    return im_eq, cy, cx
