# CS-TRD: a Cross Sections Tree Ring Detection method
Repository for the IPOL paper "CS-TRD: a Cross Sections Tree Ring Detection method". Submitted on 13/05/2023. Last Revision on 19/11/2023. 
IPOL Demo: [IPOL][link_ipol_paper].
UruDendro ImageSet: [UruDendro][link_urudendro].
ArXiv paper: [ArXiv][link_arxiv_paper].

[link_ipol_paper]: https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000390
[link_urudendro]: https://iie.fing.edu.uy/proyectos/madera/
[link_arxiv_paper]: https://doi.org/10.48550/arXiv.2305.10809

![F03d_compare.jpg](assets%2FF03d_compare.jpg)



Version 1.0
Last update: 03/12/2024
Authors: 
-	Henry Marichal, henry.marichal@fing.edu.uy
-   Diego Passarella, diego.passarella@cut.edu.uy
-   Gregory Randall, randall@fing.edu.uy


## Installation
### Conda
```bash
conda create --name ipol python==3.11
conda activate ipol
conda install -n ipol -c conda-forge geos
conda install -n ipol -c anaconda cmake 
conda install -n ipol pip
```
```bash
pip install .
```

## Examples of usage
### Import the module
```python
from cross_section_tree_ring_detection.cross_section_tree_ring_detection import TreeRingDetection

im_in = #IMAGE
args = dict(cy: 1264, cx: 1204, sigma: 1.0, th_low: 0.1, th_high: 0.2,
        hsize: 3, wsize: 3, edge_th: 0.1, nr: 1, 
        min_chain_length: 10, debug: 0, input: 'input/F02c.png',
        output_dir: './output', save_imgs: 0)
    
res = TreeRingDetection(im_in, args.cy, args.cx, args.sigma, args.th_low, args.th_high, args.hsize, args.wsize,
                            args.edge_th, args.nr, args.min_chain_length, args.debug, args.input, args.output_dir)

```
### CLI
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

## Docker Container
You can run the algorithm in a docker container.

### Pull the image
```bash
docker pull hmarichal/cstrd:v1.0
```

### Run the container
In order to run the container you need to mount a volume with the data you want to process (YOUR_DATA_FOLDER). Results 
will be stored in the mounted volume. Run the following command:
```bash
docker run -v YOUR_DATA_FOLDER:/workdir/bin/output -it hmarichal/cstrd:v1.0 / 
 python main.py --input YOUR_DATA_FOLDER/image_path --cy 1264 --cx 1204 /
 --output_dir ./output --root ./ --save_imgs 1
```

### Build the image
```bash
 docker build -f .ipol/Dockerfile . -t hmarichal/cstrd:v1.0
```

## Citation
```
@misc{marichal2023cstrd,
      title={CS-TRD: a Cross Sections Tree Ring Detection method}, 
      author={Henry Marichal and Diego Passarella and Gregory Randall},
      year={2023},
      eprint={2305.10809},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
License for th source code: [MIT](./LICENSE)





