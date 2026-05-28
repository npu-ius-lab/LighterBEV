# LighterBEV

**LiDAR Global Localization Meets Online Learning**

LighterBEV is a lightweight LiDAR global localization and loop-closure module built on BEV (Bird's-Eye View) image representations. It targets real-time deployment and online adaptation, allowing the model to update during operation and improve place recognition performance in new environments.

## Features

- Lightweight BEV-based place recognition with retrieval followed by RANSAC-based pose estimation.
- PCA-initialized Informative Compression Module (ICM) for compact local descriptors.
- Online learning pipeline with fixed-size replay buffer and descriptor refreshing.
- Evaluation pipelines for KITTI and NCLT.
- Designed to integrate into SLAM loop-closure and relocalization workflows.

## Repository Layout

- `main_pca_kitti.py`: offline training and evaluation entrypoint.
- `main_pca_nclt_OL.py`: online adaptation entrypoint for NCLT.
- `kitti_dataset.py`: KITTI dataset loading and evaluation utilities.
- `nclt_dataset.py`: NCLT dataset loading and evaluation utilities.
- `datasets/gen_bev_images.py`: helper for generating BEV images from raw point clouds.
- `RANSACCPP/`: pybind11 RANSAC extension source.

## Installation

### 1. Python environment

Python 3.8+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Build the RANSAC extension

`pybind11` is required.

```bash
cd RANSACCPP
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  rigid_ransac.cpp \
  -I/usr/include/eigen3 \
  -o rigid_ransac$(python3-config --extension-suffix)
cd ..
```

If Eigen is installed elsewhere, replace `-I/usr/include/eigen3` with the correct include path.

## Dataset Preparation

The code expects pre-generated BEV images and pose files arranged under dataset roots.

Our dataset packaging follows BEVPlace2. Download the dataset from Google Drive, unzip it, and move the files into the `data` directory:

- <https://github.com/zjuluolun/BEVPlace2>

Default roots:

- `./data/KITTI`
- `./data/NCLT`

You can override them either with command-line arguments or environment variables:

```bash
export LIGHTERBEV_KITTI_PATH=/path/to/KITTI
export LIGHTERBEV_NCLT_PATH=/path/to/NCLT
```

For raw KITTI point clouds, BEV images can be generated with:

```bash
python datasets/gen_bev_images.py \
  --vel_path /path/to/KITTI/sequences/00/velodyne \
  --bev_save_path /path/to/KITTI/00/bev_imgs
```

Dataset structure expected by the loaders:

```text
data/
  KITTI/
    00/
      bev_imgs/
    02/
      bev_imgs/
    poses/
      00.txt
      02.txt
      ...
  NCLT/
    2012-01-15/
      bev_imgs/
    2012-02-04/
      bev_imgs/
    poses/
      2012-01-15.txt
      2012-02-04.txt
      ...
```

The source code for combining this method with a SLAM system can be found at:

- <https://github.com/lbhwyy/Fastlio_Lighterbev>

## Offline Training and Evaluation

Train on KITTI:

```bash
python main_pca_kitti.py \
  --mode train \
  --batchSize 4 \
  --cacheBatchSize 64 \
  --runsPath ./runs \
  --cachePath ./cache \
  --kitti_root ./data/KITTI \
  --nclt_root ./data/NCLT
```

Test a pretrained checkpoint:

```bash
python main_pca_kitti.py \
  --mode test \
  --load_from runs/<run_name> \
  --ckpt best \
  --kitti_root ./data/KITTI \
  --nclt_root ./data/NCLT
```

Useful output arguments:

- `--globalDescPath`: directory for exported descriptors.
- `--matchResultsPath`: directory for qualitative match visualizations.

## Online Learning on NCLT

Run online adaptation:

```bash
python main_pca_nclt_OL.py \
  --mode train \
  --load_from runs/<offline_checkpoint_dir> \
  --nRuns 5 \
  --nclt_root ./data/NCLT \
  --kitti_root ./data/KITTI
```

Evaluate a specific online-learning run:

```bash
python main_pca_nclt_OL.py \
  --mode test \
  --load_from runs_online/<run_name> \
  --runsNum 0 \
  --nclt_root ./data/NCLT \
  --kitti_root ./data/KITTI
```

## Notes

- `.vscode/`, caches, run outputs, and model checkpoints are ignored by Git.
- Dataset roots are no longer hardcoded to machine-local absolute paths.
- If you plan to redistribute trained weights or derived datasets, review their respective licenses separately.

## Citation

If you use this repository in academic work, please cite the corresponding LighterBEV paper.

```bibtex
@ARTICLE{11282439,
  author={Liu, Binhong and Yang, Tao and Cao, Haoji and Fu, Shuqi and Fang, Yangwang and Yan, Zhi},
  journal={IEEE Robotics and Automation Letters},
  title={LighterBEV: LiDAR Global Localization Meets Online Learning},
  year={2026},
  volume={11},
  number={2},
  pages={1170-1177},
  doi={10.1109/LRA.2025.3641146}
}
```
