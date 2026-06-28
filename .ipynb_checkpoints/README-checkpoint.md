# [IROS 2026] A Simple Recipe for Leveraging Dual Foundation Models in Domain Generalised Semantic Segmentation

Official implementation of the IROS 2026 paper "A Simple Recipe for Leveraging Dual Foundation Models in Domain Generalised Semantic Segmentation".

> Ruoyu Guo, Jiaqi Guo, XIN KUN LIN, Maurice Pagnucco, Yang Song
> 
> University of New South Wales

## Installation

The code was tested with Python 3.10, PyTorch 2.0.1, CUDA 11.8, and MMCV 1.7.2. If you find it difficult to install the environment, we can provide a conda env package for you.

```bash
conda create -n giadapter python=3.10 -y
conda activate giadapter
pip install -r requirements.txt
pip install xformers==0.0.20
pip install mmcv-full==1.7.2 
pip install mamba_ssm==2.2.2
pip install causal_conv1d==1.4.0
```

## Pre-trained VFM & VLM Models
- Please download the pre-trained VFM and VLM models and save them in `./pretrained` folder.

  | Model | Type | Link |
  |-----|-----|:-----:|
  | DINOv2 | `dinov2_vitl14_pretrain.pth` |[download link](https://drive.google.com/file/d/1Rrl0RfU51eU8orbNVWHtNr1L3k5xhnld/view?usp=sharing)|
  | CLIP | `ViT-L-14-336px.pt` |[download link](https://drive.google.com/file/d/1s00ofvxn0NCVVgnycXd2wUx4Gs53O6mj/view?usp=sharing)|
  | EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B.pt` |[download link](https://drive.google.com/file/d/1mQJ1zc_YLt7qAbaAET4-2EGNtIp2I6eB/view?usp=sharing)|
  | SIGLIP | `siglip_vitl16_384.pth` |[download link](https://drive.google.com/file/d/1PezEbwpqlasSYH2KPtU3aUD4hCzk9uE-/view?usp=sharing)|


## Checkpoints

Please download GI-Adapter checkpoints from [download link](https://drive.google.com/drive/folders/1EcKBnLFryA48yUw3IQQWKW7r2WRFGPw4?usp=drive_link) and save them in the `./work_dirs_d` folder.

| Model name | Pretrained | Trained on |
| --- | --- | --- |
| clip-gta | CLIP | GTA |
| clip-city | CLIP | Cityscapes |
| clip-syn | CLIP | SYNTHIA |
| eva-gta | EVA02-CLIP | GTA |
| eva-city | EVA02-CLIP | Cityscapes |
| eva-syn | EVA02-CLIP | SYNTHIA |
| siglip-gta | SigLIP | GTA |
| siglip-city | SigLIP | Cityscapes |
| siglip-syn | SigLIP | SYNTHIA |


## Datasets
- After downloading the datasets, edit the data configs in ```/GI-Adapter/configs/_base_/datasets``` following your environment.
  
  ```python
  src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  ```

- **Data Preprocessing:** Finally, run the following script to convert the label IDs. Before running, open the script and modify the dataset paths to match your local directories.
    ```shell
        sh ./tools/convert_datasets/convert.sh
    ```

- Note that RAND_CITYSCAPES is SYNTHIA. The final folder structure should look like this:
```
GI-Adapter
├── ...
├── pretrained
│   ├── dinov2_vitl14_pretrain.pth
│   ├── EVA02_CLIP_L_336_psz14_s6B.pt
│   ├── siglip_vitl16_384.pth
│   ├── ViT-L-14-336px.pt
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── bdd100k
│   │   ├── images
│   │   │   ├── 10k
│   │   │   │    ├── train
│   │   │   │    ├── val
│   │   ├── labels
│   │   │   ├── sem_seg
│   │   │   │    ├── masks
│   │   │   │    │    ├── train
│   │   │   │    │    ├── val
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscapes_trainIdLabel
│   │   ├── half
│   │   │   ├── val_img
│   │   │   ├── val_label
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── ACDC
│   │   ├── gt
│   │   │   ├── fog
│   │   │   ├── night
│   │   │   ├── rain
│   │   │   ├── snow
│   │   ├── rgb_anno
│   │   │   ├── fog
│   │   │   ├── night
│   │   │   ├── rain
│   │   │   ├── snow
│   ├── RAND_CITYSCAPES
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

## Training
```
python train.py configs/[TRAIN_CONFIG]
```

## Evaluation

```
python test.py configs/[TEST_CONFIG] work_dirs_d/[MODEL-NAME] --eval mIoU
```

## Citation

If you find our code helpful, please cite our paper:

```bibtex
@inproceedings{guo2026giadapter,
  title     = {A Simple Recipe for Leveraging Dual Foundation Models in Domain Generalised Semantic Segmentation},
  author    = {Guo, Ruoyu and Guo, Jiaqi and Lin, Xinkun and Pagnucco, Maurice and Song, Yang},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## Acknowledgements

This project is based on the following open-source projects. We thank the authors for sharing their code.

- [MFuser](https://github.com/devinxzhang/MFuser)
