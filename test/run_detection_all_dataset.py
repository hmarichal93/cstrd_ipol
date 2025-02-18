import os
import pandas as pd
from pathlib import Path
from lib.io import get_path
from fpdf import FPDF
import glob
from natsort import natsorted
from tqdm import tqdm

def generate_pdf(path):
    pdf = FPDF()
    pdf.set_font('Arial', 'B', 16)

    resultado_final = "output.png"
    figures = glob.glob(f"{path}/**/{resultado_final}", recursive=True)
    idx = 0
    for fig in tqdm(natsorted(figures)):
        idx += 1
        splitted = fig.split("/")
        disco = splitted[-2]
        disco = disco.replace("_", "")

        fig_path = ""
        for sub_path in splitted[:-1]:
            fig_path += sub_path + "/"

        # fig2 = fig_path + "preprocessing_output.png"
        fig3 = fig_path + "chains.png"
        #fig4 = fig_path + "connect.png"
        #fig5 = fig_path + "postprocessing.png"
        fig6 = fig_path + "output.png"
        x, y = 0, 40
        height = 180

        for fig in [fig6]:
            pdf.add_page()
            pdf.cell(0, 0, disco)
            pdf.image(fig, x, y, h=height)

        #pdf.add_page()

    pdf.output(f"{path}/summary_ipol.pdf", 'F')

def main(root_database = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV1",  results_path="/data/maestria/resultados/cstrd_round_3_1_table_9_2"):

    metadata_filename = Path(root_database) / 'dataset_ipol.csv'
    images_dir = Path(root_database) / "images/segmented"
    results_path = Path(results_path)
    results_path.mkdir(exist_ok=True)

    metadata = pd.read_csv(metadata_filename)
    for idx in range(metadata.shape[0]):

        row = metadata.iloc[idx]
        name = row.Imagen


        img_res_dir = (results_path / name)

        img_res_dir.mkdir(exist_ok=True)

        img_filename = images_dir / f"{name}.png"
        cy = row.cy
        cx = row.cx
        sigma = row.sigma
        sigma = 3
        if (img_res_dir / "labelme.json").exists():
            continue

        command = f"python main.py --input {img_filename} --sigma {sigma} --cy {cx} --cx {cy}  --root ./ --output_dir" \
                  f" {img_res_dir} --hsize 1500 --wsize 1500 --save_imgs 1"
        command = f"python main.py --input {img_filename} --sigma {sigma} --cy {cx} --cx {cy}  --root ./ --output_dir" \
                  f" {img_res_dir} --hsize 1500 --wsize 1500"

        print(command)
        os.system(command)

    #generate_pdf(results_path)



if __name__=='__main__':
    main()




