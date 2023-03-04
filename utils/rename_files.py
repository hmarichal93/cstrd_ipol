from pathlib import Path
import os

class DiskNameConvertion:
    convertion_dictionary = {'ab': 'a', 'a': 'b', 'a_cut': 'c', 'b_cut': 'd', 'b': 'e'}
    def __init__(self, name):
        self.letter = name[0]
        nros = []
        rest_list = []
        for key in DiskNameConvertion.convertion_dictionary.keys():
            if key in name[1:].lower():
                nro, rest_without_key = name[1:].lower().split(key)
                if len(nro) <= 2:
                    self.tree_number = nro
                    nros.append(nro)
                    rest_list.append(key+rest_without_key)

        rest = sorted(rest_list)[0]
        self.tree_number = nros[rest_list.index(rest)]
        self.correlative_letter = self.rest_convertion(rest)

    def rest_convertion(self, rest):

        keys_in_string = [key for key in DiskNameConvertion.convertion_dictionary.keys() if key in rest.lower()]
        if len(keys_in_string) > 1:
            code = list(DiskNameConvertion.convertion_dictionary.keys())[2]
            if code in keys_in_string:
                letter = DiskNameConvertion.convertion_dictionary[code]
            else:
                code = list(DiskNameConvertion.convertion_dictionary.keys())[3]
                if code in keys_in_string:
                    letter = DiskNameConvertion.convertion_dictionary[code]
                else:
                    code = list(DiskNameConvertion.convertion_dictionary.keys())[0]
                    if code in keys_in_string:
                        letter = DiskNameConvertion.convertion_dictionary[code]
                    else:
                        print(rest)
                        raise
        else:
            letter = DiskNameConvertion.convertion_dictionary[keys_in_string[0]]

        return letter

import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    #images_path = Path('/data/maestria/resultados/ipol_refactoring_resize_experiment_1500')
    images_path = Path(args.path)
    dst_images_path = Path('/data/maestria/database_new_nomenclature/ground_truth_processed/annotations')
    for disk_path in images_path.rglob('labelme.json'):
        if 'fx' not  in disk_path.parent.name:
            continue
        name = disk_path.parent.name
        # fileds = name.split('_')
        # creator = fileds[-1]
        # name =fileds[0]
        # for campo in fileds[1:-1]:
        #     name+=f"_{campo}"
        new_name = name#DiskNameConvertion(name)
        #new_name = f"{new_name.letter}{int(new_name.tree_number):02d}{new_name.correlative_letter}"
        print(f"{name}-->{new_name}")
        os.system(f'cp {disk_path} {str(dst_images_path)}/{new_name}_{images_path.name}.json')

        # #
        # for gt_path in Path('./ground_truth').rglob(f'{name}*'):
        #     gt_name,ext = gt_path.name.split('.')
        #     gt_new_name = gt_path.parent / f"{new_name}.{ext}"
        #     os.system(f"mv {gt_path} {gt_new_name}")

