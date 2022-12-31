import numpy as np
import cv2
import time
from PIL import Image
import numpy as np

import lib.chain_v4 as ch

def resize_image(img: np.array, newsize):
    pil_img = Image.fromarray(img)
    #Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    flag = Image.ANTIALIAS
    #qqqqqqqqflag = Image.Resampling.LANCZOS
    pil_img = pil_img.resize(newsize, flag)
    np_img = np.array(pil_img)
    return np_img
def main(datos):
    M, N, img, SAVE_PATH = datos['M'], datos['N'], datos['img'], datos['save_path']
    t0 = time.time()

    imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resize.
    if len(datos['config']['resize'])>0:
        hnew, wnew = datos['config']['resize']
        hscale = hnew / M
        wscale = wnew / N
        imageGray = resize_image(imageGray,( hnew, wnew) )
        img = resize_image(img, (hnew, wnew))
        centro = datos['centro']
        datos['centro'] = (int(centro[0] * wscale), int(centro[1] * hscale))
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
    datos['img'] = img
    print(f'Preprocessing: {tf-t0:.1f} seconds')

    return 0