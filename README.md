# LighterBEV

**LiDAR Global Localization Meets Online Learning**

LighterBEV is a lightweight LiDAR global localization and loop-closure module built on **BEV (Bird’s-Eye View) image representations**. It is designed for **real-time deployment** and **post-deployment online adaptation**, enabling a robot to **update model parameters during operation** and **improve place recognition performance in new environments**. 

---

## News and Highlights

- **Lightweight BEV-based place recognition + pose estimation**: global descriptor retrieval followed by **RANSAC-based** relative pose estimation.
- **PCA-initialized Informative Compression Module (ICM)**: compresses local features via a trainable linear projection **initialized by PCA**, preserving discriminative structure and rotation equivariance.
- **Online learning for LiDAR global localization**: maintains a fixed-size buffer with **reservoir adding**, **informative sampling**, and **feature refreshing** for efficient continual adaptation under streaming constraints.
- **Designed to integrate with SLAM**: online updates occur during the SLAM process, supporting loop closure/relocalization pipelines.

---

## Method Overview

### BEV Global Localization Pipeline

Given a LiDAR scan, we generate a BEV image and extract:

1. **Global descriptor** for retrieval (place recognition).
2. **Compact local descriptors** for geometric verification and pose estimation.

The system follows a two-stage localization routine:
**(i) place recognition** → **(ii) pose estimation with RANSAC**. 

### LighterREM Encoder (REM + ICM)

LighterBEV adopts a rotation-equivariant feature encoder and introduces an **Informative Compression Module (ICM)** for efficient dimensionality reduction. 

**ICM core idea (PCA initialization, trainable refinement):**
A set of local features is sampled, PCA is applied to obtain a projection matrix and mean vector, then the projection is made trainable and optimized end-to-end:

- Trainable parameters are initialized by PCA,
- Updated by backpropagation during training.

### Online Learning Mechanism

The online learning pipeline maintains an **online buffer** and repeats:

- Compute descriptor for new frame
- Add to buffer (reservoir)
- Sample informative triplets (hard negatives)
- Update network
- Refresh buffer descriptors periodically 

---

## Dependencies

- Python **3.8+**
- PyTorch (CUDA version should match your system)
- faiss, numpy, opencv-python, imgaug, tqdm, tensorboardX, h5py, scikit-learn, matplotlib, pympler
- (Optional) ROS/SLAM stack for integration

## Quickstart

This section walks you through a minimal end-to-end workflow:
**(1) prepare BEV images → (2) compile RANSAC → (3) offline train/eval → (4) online adaptation**.

### Step 1. Prepare Datasets

### Step 2. Build rigid_ransac

pybind11 is needed

On Ubuntu 20.04, compile the RANSAC extension with:

```bash
cd RANSACCPP
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  rigid_ransac.cpp \
  -I/usr/include/eigen3 \
  -o rigid_ransac$(python3-config --extension-suffix)
```

If Eigen is installed in a different location, replace -I/usr/include/eigen3 with your actual include path.

### Step 3. Offline Training and evaluation (KITTI → NCLT Evaluation)

python main_pca_kitti.py
--mode train
--batchSize 4
--cacheBatchSize 64
--runsPath ./runs/
--cachePath ./cache/

Offline Evaluation
python main_pca_kitti.py
--mode test
--load_from runs/xx_xx-xx-xx
--ckpt best

--load_from: directory for pre-trained checkpoints

### Step 4. Online Training and evaluation

Online Learning (Post-deployment Adaptation)
Online Training

python main_pca_nclt_OL.py
--mode train
--load_from runs_online/xx_xx-xx-xx
--nRuns 5
Notes:

--load_from: a pretrained checkpoint folder (i.e. trained on KITTI).

--nRuns: run multiple times and average results to reduce randomness.

Testing (Evaluate a Specific Run)
python main_pca_nclt_OL.py
--mode test
--load_from runs_online/xx_xx-xx-xx
--runsNum 0

Citation

If you use this repository in academic work, please cite the corresponding LighterBEV paper (add BibTeX here).

@article{lighterbev2025,
title   = {LighterBEV: LiDAR Global Localization Meets Online Learning},
author  = {Liu, Binhong and Yang, Tao and Cao, Haoji and Fu, Shuqi and Fang, Yangwang and Yan, Zhi},
journal = {IEEE Robotics and Automation Letters},
year    = {2025}
}

```

```
