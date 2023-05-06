# CS-TRD: a Tree Ring Detection method for tree cross
sections

Repository for the IPOL paper "CS-TRD: a Tree Ring Detection method for tree cross
sections"
[IPOL][link_ipol_paper].

[link_ipol_paper]: https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000390

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

## Usage
```bash
python main.py --input IMAGE_PATH --cx CX --cy CY 
  --output_dir OUTPUT_DIR --root REPO_ROOT_DIR
```

## Automatic center detection
Detecting pith center automatically can be done using software from IPOL paper "Ant Colony Optimization for Estimating Pith Position on Images of Tree Log Ends" [IPOL][link_ipol_pith_paper].

[link_ipol_pith_paper]: https://www.ipol.im/pub/art/2022/338/?utm_source=doi

## Citation


## License
License for th source code: [MIT](./LICENSE)





