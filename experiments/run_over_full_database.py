import os
import pandas as pd

from lib.io import get_path

if __name__=='__main__':
    database_path = get_path('data')
    metadata_filename = get_path('metadata') / 'dataset_ipol.csv'
    results_path = get_path('results') / "ipol_refactoring"
    results_path.mkdir(exist_ok=True)

    metadata = pd.read_csv(metadata_filename)
    for idx in range(metadata.shape[0]):
        # if idx not in [10,15,20]:
        #     continue
        row = metadata.iloc[idx]
        name = row.Imagen
        if name not in 'F03d':
            continue
        img_res_dir = (results_path / name)

        img_res_dir.mkdir(exist_ok=True)
        dt_file = img_res_dir / "labelme.json"
        if dt_file.exists():
            continue

        img_filename = database_path / f"{name}.png"
        cy = row.cy
        cx = row.cx
        sigma = row.sigma
        command = f"python main.py --input {img_filename} --sigma {sigma} --cy {cx} --cx {cy}  --root ./ --output_dir" \
                  f" {img_res_dir}"
        print(command)
        os.system(command)

    # #metrics
    # os.system(f"python utils/rename_files.py")
    # os.system(f"python lib/metrics_new_nomenclature.py")
    #
    # #group chains
    # os.system(f"python utils/generate_pdf_results_union_chains.py --root_dir {results_path}")





