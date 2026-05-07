# LighterBEV

基于激光点云 BEV 图像的轻量化地点识别与定位实现，包含 KITTI 与 NCLT 数据集上的离线训练、全局/局部特征提取、回环检测，以及支持在线持续学习的实验脚本。
- LiDAR 点云转 BEV（200×200）灰度图，使用 `REIN_PCA` 提取局部特征并经 PCA+NetVLAD 聚合为全局描述子。
- 覆盖多种评价模式：纯地点检索（PR）、全局定位（RANSAC+姿态优化）、回环检测 PR 曲线，以及 NCLT 跨序列定位。
- 提供在线学习流程（重放缓存 + 随机遮挡增强），便于 OCL 研究。
- 关键脚本：`main_pca_kitti.py`（离线训练/评估），`main_pca_nclt_OCL.py`（在线学习），`datasets/gen_bev_images.py`（点云转 BEV），`eval/loop_closure.py`（可视化/统计回环结果）。

## 环境依赖
- Python 3.8+，CUDA 环境建议与本地 PyTorch 版本一致。
- PyTorch、faiss、numpy、opencv-python、imgaug、tqdm、tensorboardX、h5py、scikit-learn、matplotlib、pympler。


## 数据准备
1) 目录组织（KITTI 示例，更多见 `datasets/data.md`）：
```
datasets/
  └─ KITTI/
      ├─ 00/
      │   └─ bev_imgs/        # 由 velodyne 点云生成的 BEV PNG
      ├─ 02/
      │   └─ bev_imgs/
      ├─ 05/
      │   └─ bev_imgs/
      ├─ 06/
      │   └─ bev_imgs/
      ├─ 08/
      └─ poses/seq.txt        # Nx12, 3/7/11 列为 x/y/z 平移
```
NCLT 结构类似：`datasets/NCLT/<seq>/bev_imgs/` 与 `datasets/NCLT/poses/<seq>.txt`。



## 编译 RANSACCPP/rigid_ransac
在 Ubuntu20.04 下可用一条命令完成编译（确保已安装 `pybind11` 和 Eigen3）：
```bash
cd RANSACCPP
c++ -O3 -Wall -shared -std=c++17 -fPIC   $(python3 -m pybind11 --includes)   rigid_ransac.cpp   -I/usr/include/eigen3   -o rigid_ransac$(python3-config --extension-suffix)
```
如 Eigen 路径不同，将 `-I/usr/include/eigen3` 替换为实际 include 路径。


## 训练与评估
### 离线训练（KITTI->NCLT 评测）
```bash
python main_pca_kitti.py \
  --mode train \
  --batchSize 4 \
  --cacheBatchSize 64 \
  --runsPath ./runs/ \
  --cachePath ./cache/
```

### 离线评估/特征导出
```bash
python main_pca_kitti.py \
  --mode test \
  --load_from runs/xx_xx-xx-xx \
  --ckpt best
```
默认同时运行 KITTI（PR / global localization / loop closure）与 NCLT（loop closure / cross localization）。全局描述子保存到 `global_descripors_ol/`，匹配可视化保存到 `out_imgs/`。

### 在线学习
```bash
python main_pca_nclt_OL.py
  --mode train
  --load_from runs_online/xx_xx-xx-xx
  --nRuns 5
```
--mode train --load_from KITTI上预训练的权重文件夹 --nRuns 跑nRuns取平均值减小随机性
### 测试在线学习效果
```bash
python main_pca_nclt_OL.py
  --mode test
  --load_from runs_online/xx_xx-xx-xx
  --runsNum 0

```
--mode train --load_from KITTI上预训练的权重文件夹 --runsNum评估哪一次run的结果
脚本会按序流式读取帧，利用 `buffer.Buffer` 进行重放采样并实时更新模型。


