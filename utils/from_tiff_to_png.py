import cv2
import numpy as np
import glob
from lib.io import get_path

def main():
    dataset = get_path('data') / "ipol_tiff"


    images_files = glob.glob(f"{str(dataset)}/*.tif")
    images_files += glob.glob(f"{str(dataset)}/*.jpg")

    for filename in images_files:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640,640))
        output_filename = filename[:-4] + '.png'

        cv2.imwrite(output_filename, img)




    return

if __name__=="__main__":
    main()

