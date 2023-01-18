import numpy as np
import argparse
import cv2

def mannually(filename):
    f = open(filename, "r")
    string = f.readlines()[0]
    y, x = string.replace("[", "").replace("]", "").split(",")
    f.close()
    return y,x

def automatic(filename):
    image_name, _ = filename.split(".")
    arr = np.loadtxt(f"{image_name}.csv",delimiter=",", dtype=str)
    y = arr[1][0]
    x = arr[1][1]
    return y,x

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--type", type=int, required=True, default=0)
    args = parser.parse_args()
    if args.type == 0:
        y,x = mannually(args.input)
    else:
        y,x = automatic(args.input)

    print(f"{int(y)}")
    print(f"{int(x)}")
