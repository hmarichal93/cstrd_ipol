import os
import pandas as pd

from lib.io import get_path
import argparse
from pathlib import Path



if __name__=='__main__':
    database_path = get_path('data')
    metadata_filename = get_path('metadata') / 'dataset_ipol.csv'
    metadata = pd.read_csv(metadata_filename)

    size = 1500
    #sigma = 2.0
    for sigma in [2.0]:
        results_path = get_path('results') / f"ipol_refactoring_resize_experiment_{size}_{sigma}"
        path_data = str(results_path).replace(".", "_")
        results_path = Path(path_data)
        results_path.mkdir(exist_ok=True)
        for idx in range(metadata.shape[0]):
            # if idx not in [0]:
            #     continue
            row = metadata.iloc[idx]
            name = row.Imagen
            img_res_dir = (results_path / name)

            img_res_dir.mkdir(exist_ok=True)

            dt_file = img_res_dir / "labelme.json"
            if dt_file.exists():
                continue

            img_filename = database_path / f"{name}.png"
            cy = row.cy
            cx = row.cx
            #sigma = row.sigma
            command = f"python main.py --input {img_filename} --sigma {sigma} --cy {cx} --cx {cy}  --root ./ --output_dir" \
                      f" {img_res_dir}"
            print(command)
            os.system(command)

        #metrics
        path_data = str(results_path).replace(".", "_")
        os.system(f"python utils/rename_files.py --path {path_data}")
        os.system(f"python lib/metrics_new_nomenclature.py --creator {str(results_path.name).replace('.','_')}")

        #group chains
        #os.system(f"python utils/generate_pdf_results_union_chains.py --root_dir {results_path}")





