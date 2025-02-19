## SaliencyI2PLoc

### Task target

This paper focuses on achieving the fusion of images and point clouds to enable coarse visual localization of a single image within a pre-built point cloud map.

### Installation

```bash
git clone https://github.com/whu-lyh/SaliencyI2PLoc.git --recursive
cd scripts
bash install.sh
```

> You may required to change the coding manner of sh files using `sed -i "s/\r//" *.sh` to avoid the file unrecognition.

+ both pytorch1.13.1-cuda11.6 and pytorch2.1.2-cuda12.1 works

The [model weights](https://drive.google.com/drive/folders/1FdUSGHTEpjTKHDb-uXQEuvKA4-kePeBs?usp=drive_link) and tha [datasets](https://drive.google.com/drive/folders/1Ohw0Ha-yGJL8h5rAk7P3UhPuLabRBbXq?usp=drive_link) could be downloaded from GoogleDrive. The pretrained models of ResNet and ViT used in our job could be download at [here](https://drive.google.com/drive/folders/1Om-keiYXQWDcdqtzPa54WbqPZFbJg461?usp=drive_link).

### Train

```bash
cd scripts
bash train.sh
```

The configuration information will be loaded all in once from the `CrossModalityRetrieval.yaml` style file, including the optimizer, scheduler, dataset, model and other configuration.

### Test

```bash
cd scripts
bash test.sh
```

Adjust the test data sequences that you want to test at `/config/dataset_configs` folder.

### Models

The details of the used model can be found in [Architectures.md](./docs/Architectures.md).

### Datasets

The details of the used datasets can be found in [Datasets.md](./docs/Datasets.md).

### Citation

If you find our work is useful to yours, please cite our paper.
```
@article{LI2025103015,
title = {SaliencyI2PLoc: Saliency-guided image–point cloud localization using contrastive learning},
journal = {Information Fusion},
volume = {118},
pages = {103015},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103015}
}
```
