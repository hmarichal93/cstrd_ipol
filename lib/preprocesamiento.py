import numpy as np
import cv2
import time

import lib.chain_v4 as ch

def main(datos, kernel_size = 21):
    M, N, img, SAVE_PATH, sigma = datos['M'], datos['N'], datos['img'], datos['save_path'], datos['sigma']
    t0 = time.time()
    #imageGray = utils.rgbToluminance(img)
    imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_eq = equalize_adapthist(np.uint8(imageGray), clip_limit=0.03)
    mask = np.where(imageGray==255,1,0)
    img_eq = imageGray.copy()
    img_eq[mask>0] = np.mean(img_eq[mask==0])
    clahe = cv2.createCLAHE(clipLimit=10)
    img_blur = clahe.apply(img_eq)
    img_blur = cv2.bitwise_not(img_blur)
    img_blur[mask>0] = 255
    ch.visualizarCadenasSobreDisco(
        [], img_blur,f"{SAVE_PATH}/preprocessing_output.png", labels=False, gris=True, color=True
    )

    tf = time.time()
    datos['img_prep'] = img_blur
    datos['tiempo_preprocessing'] = tf-t0
    print(f'Preprocessing: {tf-t0:.1f} seconds')

    return 0