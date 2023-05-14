# CS-TRD: a Cross Sections Tree Ring Detection method
Repository for the IPOL paper "CS-TRD: a Cross Sections Tree Ring Detection method"
[IPOL][link_ipol_paper].

[link_ipol_paper]: https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000390

Version 1.0
Last update: 13/05/2023
Authors: 
-	Henry Marichal, henry.marichal@fing.edu.uy
-   Diego Passarella, diego.passarella@cut.edu.uy
-   Gregory Randall, randall@fing.edu.uy

## Get started

#### 1. Folders
All the python source files are in lib/ folder. Dependencies are in externas/ folder. 
Default configuration files are in config/ folder. 

    
Algorithm 1 is implemented in the file **main.py**

Algorithm 2 is implemented in the file **lib/preprocessing.py**

Canny-Devernay implementation [[10](https://www.ipol.im/pub/art/2017/216/)] is in **externas/devernay_1.0**. Python
wrapper is in **lib/canny_devernay_edge_detector.py**

Algorithm 5 is implemented in the file **lib/filter_edges.py** 

Algorithm 7 is implemented in the file **lib/sampling.py**

Algorithm 8 is implemented in the file **lib/connect_chains.py**

Algorithm 19 is implemented in the file **lib/postprocessing.py**
## Installation
```bash
apt-get update && apt-get install -y $(cat .ipol/packages.txt) &&
  rm -rf /var/lib/apt/lists/* 
```

```bash
pip3 install --no-cache-dir -r requirements.txt
```
```bash
cd ./externas/devernay_1.0 && make clean && make
```

## Examples of usage

Here some examples of usage:
```bash
python main.py --input IMAGE_PATH --cx CX --cy CY 
  --output_dir OUTPUT_DIR --root REPO_ROOT_DIR
```
```bash
python main.py --input input/F02c.png --cy 1264 --cx 1204  --output_dir ./output --root ./
```
If you want to run the algorithm generating intermediate results you can use the flag --save_imgs

```bash
python main.py --input input/F02c.png --cy 1264 --cx 1204  --output_dir ./output --root ./ --save_imgs 1
```

## Automatic center detection
Detecting pith center automatically can be done using software from IPOL paper "Ant Colony Optimization for Estimating Pith Position on Images of Tree Log Ends" [IPOL][link_ipol_pith_paper].

[link_ipol_pith_paper]: https://www.ipol.im/pub/art/2022/338/?utm_source=doi

## Citation


## License
License for th source code: [MIT](./LICENSE)





