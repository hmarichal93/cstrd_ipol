import numpy as np
from pathlib import Path
import os

if __name__ == '__main__':
    data_dir = Path('/data/maestria/resultados/ipol2')
    for file in data_dir.glob("*"):
        name = file.name
        name,_ = name.split('_')
        #os.system(f"mv {str(file)} {str(file.parent)}/{name}.json")