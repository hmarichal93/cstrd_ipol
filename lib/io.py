import os
import json
from pathlib import Path
import cv2
import time

class Disk:
    def __init__(self, cy, cx, output_dir, image, config, start_time):
        self.center_y = cy
        self.center_x = cx
        self.save_dir = output_dir
        self.img = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.debug = config.get('debug', False)
        self.data_dic = {}
        self.nr = config['Nr']
        self.resize = config['resize']
        self.th_low = config['th_low']
        self.th_high = config['th_high']
        self.min_chain_lenght = config['min_chain_lenght']
        self.edge_th = config['edge_th']
        self.sigma = config['sigma']
        self.devernay_path = config.get('devernay_path')
        self.start_time = start_time

    def print_time(self):
        for key in self.data_dic:
            print(f"{key} {self.data_dic[key]['time']}")
        print(f"total time {time.time()-self.start_time}")

def load_image(image_name):
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_config(default=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_json(f"{dir_path}/../config/default.json") if default else load_json(f"{dir_path}/../config/general.json")
def BuildingContext(image_name, cy, cx, working_dir, output_dir):
    """
    Load image
    :param image_name: paht where image is
    :param cy: pith'y coodinate
    :param cx: pith'x coodinate
    :param working_dir: working directory
    :param output_dir: directory where save results
    :return: void
    """

    config = load_json(f"{working_dir}/config/general.json")
    img = load_image(image_name)
    disk = Disk(cy, cx, output_dir, img, config, time.time())

    return disk

def load_json(filepath: str) -> dict:
    """
    Load json utility.
    :param filepath: file to json file
    :return: the loaded json as a dictionary
    """
    with open(str(filepath), 'r') as f:
        data = json.load(f)
    return data


def write_json(dict_to_save: dict, filepath: str) -> None:
    """
    Write dictionary to disk
    :param dict_to_save: serializable dictionary to save
    :param filepath: path where to save
    :return: void
    """
    with open(str(filepath), 'w') as f:
        json.dump(dict_to_save, f)


def get_path(*args):
    """
    Return the path of the requested dir/s
    Possible arguments: "data", "bader_data", "training", "results"
    :return: Path/s
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    paths = load_json(f"{dir_path}/../paths_config.json")
    hostname = 'henry-workstation'
    assert hostname in paths.keys(), "Current host: {}, Possible hosts: {}".format(hostname, paths.keys())
    assert all([arg in paths[hostname].keys() for arg in args]), "Args must be in {}".format(paths[hostname].keys())
    paths = tuple([Path(paths[hostname][arg]) for arg in args])
    return paths[0] if len(paths) == 1 else paths


