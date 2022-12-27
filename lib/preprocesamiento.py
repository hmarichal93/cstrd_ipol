import numpy as np
import cv2
from skimage.exposure import equalize_adapthist
import time

from lib import utils
import lib.chain_v4 as ch

def main(datos, kernel_size = 21):
    M, N, img, SAVE_PATH, sigma = datos['M'], datos['N'], datos['img'], datos['save_path'], datos['sigma']
    t0 = time.time()
    imageGray = utils.rgbToluminance(img)
    img_eq = equalize_adapthist(np.uint8(imageGray), clip_limit=0.03)
    img_blur = img_eq#cv2.GaussianBlur(img_eq, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    #cv2.imwrite(f"{SAVE_PATH}/preprocessing_output.png", img_blur*255)
    ch.visualizarCadenasSobreDisco(
        [], img_blur*255,f"{SAVE_PATH}/preprocessing_output.png", labels=False, gris=True, color=True
    )

    tf = time.time()
    datos['img_prep'] = img_blur
    datos['tiempo_preprocessing'] = tf-t0
    print(f'Preprocessing: {tf-t0:.1f} seconds')

    return 0