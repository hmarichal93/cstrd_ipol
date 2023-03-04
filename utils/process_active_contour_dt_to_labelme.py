from pathlib import Path
import numpy as np

from lib.io import write_json

def ring_list_2_labelme_json(rings_array):
    """
    chain list: chain list
    """

    labelme_json = {"imagePath":None, "imageHeight":None,
                    "imageWidth":None, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData": None, 'exec_time(s)':None}

    for idx, chain in enumerate(rings_array):
        ring = {"label":str(idx+1)}
        ring["points"] = chain
        ring["shape_type"]="polygon"
        ring["flags"]={}
        labelme_json["shapes"].append(ring)

    return labelme_json
if __name__=='__main__':
    active_contour_dt_dir = Path('/data/maestria/datasets/CONTORNOS_ACTIVOS/dt')
    for file in active_contour_dt_dir.glob("*gt.txt"):
        labelme = {}
        name,_ = file.name.split(".")

        #load data
        f = open(str(file), "r")
        rings = []
        ring = []
        for x in f:
            print(x)
            if 'R' in x:
                rings.append(ring)
                ring = []
            else:
                y,x = x.replace("\n", "").split(" ")
                ring.append([int(y), int(x)])
        f.close()
        rings.pop(0)
        labelme = ring_list_2_labelme_json(rings)
        write_json( labelme, str(file.parent / f"{name}_labelme.json"))
        break
