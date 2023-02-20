import numpy as np
import pandas as pd

from utils.rename_files import DiskNameConvertion

def main():
    filename = '/data/maestria/database_new_nomenclature/gt_desc.csv'
    df = pd.read_csv(filename)
    for idx, row in df.iterrows():
        old_name = row.imagen

        if 'fx' not in old_name:
            new_name = DiskNameConvertion(old_name)
            new_name = f"{new_name.letter}{int(new_name.tree_number):02d}{new_name.correlative_letter}"
        else:
            new_name = old_name


        print(f'old {old_name} new {new_name}\n\n\n')
        dictinary = row.to_dict()
        dictinary['imagen'] = new_name
        df.loc[idx] = dictinary
        print(f'old {df.loc[idx].imagen } new {new_name}\n\n\n')

    df.to_csv('/data/maestria/database_new_nomenclature/gt_desc_2.csv')




    return

if __name__=="__main__":
    main()