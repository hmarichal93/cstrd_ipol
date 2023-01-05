import numpy as np
import argparse
import cv2

def main(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x,y = np.where(img>0)
    print(f"{np.mean(y):.0f}")
    print(f"{np.mean(x):.0f}")
    return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    main(args.input)
