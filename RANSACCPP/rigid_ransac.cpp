#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <tuple>
#include <cmath>

namespace py = pybind11;

Eigen::Matrix3d svdICP(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) {
    // 计算均值，并保持为列向量 (2, 1)
    Eigen::Vector2d mean_src = src.colwise().mean();
    Eigen::Vector2d mean_dst = dst.colwise().mean();
    Eigen::MatrixXd src_norm = src.rowwise() - mean_src.transpose();
    Eigen::MatrixXd dst_norm = dst.rowwise() - mean_dst.transpose();
    
    // 计算协方差矩阵
    Eigen::Matrix2d mat_s = src_norm.transpose() * dst_norm;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat_s, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // 获取 U 和 V^T
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V_T = svd.matrixV().transpose();
    
    // 计算旋转矩阵
    Eigen::Matrix2d temp = U * V_T;
    double det = temp.determinant();
    Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
    S(1, 1) = det;
    
    // 修复旋转矩阵计算
    Eigen::Matrix2d mat_r = V_T.transpose() * S * U.transpose();

    // 计算平移向量
    Eigen::Vector2d translation = mean_dst - mat_r * mean_src;

    // 构建 3x3 刚体变换矩阵
    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    transform.block<2, 2>(0, 0) = mat_r;
    transform.block<2, 1>(0, 2) = translation;

    return transform;
}

// rigidRansac: 基于RANSAC的刚体变换估计
std::tuple<Eigen::Matrix3d, std::vector<bool>, int> rigidRansac(
    const Eigen::MatrixXd& points1, 
    const Eigen::MatrixXd& points2, 
    int iters = 1000, 
    double threshold = 0.5) {
    
    if (points1.rows() != points2.rows()) {
        throw std::invalid_argument("points1 and points2 must have the same number of rows.");
    }
    
    int max_inliers = 0;
    std::vector<bool> best_mask(points1.rows(), false);
    Eigen::Matrix3d best_transform = Eigen::Matrix3d::Identity();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points1.rows() - 1);
    
    for (int i = 0; i < iters; ++i) {
        // 随机选取两个点对
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        while (idx2 == idx1) {
            idx2 = dis(gen);
        }
        
        // 构建 2x2 矩阵用于 SVD
        Eigen::MatrixXd x(2, 2), y(2, 2);
        x.row(0) = points1.row(idx1);
        x.row(1) = points1.row(idx2);
        y.row(0) = points2.row(idx1);
        y.row(1) = points2.row(idx2);
        
        // 调用 svdICP 计算变换矩阵
        Eigen::Matrix3d rot_mat = svdICP(x, y);
        
        
        // 应用变换矩阵
        Eigen::MatrixXd y_hat = (points1 * rot_mat.block<2, 2>(0, 0).transpose()).rowwise() 
                                + rot_mat.block<2, 1>(0, 2).transpose();
        
        // 计算误差
        Eigen::MatrixXd err = (y_hat - points2).rowwise().norm();
        
        // 共识集计数
        int consensus_num = 0;
        std::vector<bool> mask(points1.rows(), false);
        for (int j = 0; j < err.rows(); ++j) {
            if (err(j) < threshold) {
                mask[j] = true;
                consensus_num++;
            }
        }
        
        // 更新最优结果
        if (consensus_num > max_inliers) {
            max_inliers = consensus_num;
            best_mask = mask;
            best_transform = rot_mat;
        }
    }
    
    // 最后用共识集重新计算变换矩阵
    Eigen::MatrixXd points1_c(max_inliers, 2);
    Eigen::MatrixXd points2_c(max_inliers, 2);
    int idx = 0;
    for (int i = 0; i < points1.rows(); ++i) {
        if (best_mask[i]) {
            points1_c.row(idx) = points1.row(i);
            points2_c.row(idx) = points2.row(i);
            idx++;
        }
    }
    
    best_transform = svdICP(points1_c, points2_c);
    
    return std::make_tuple(best_transform, best_mask, max_inliers);
}

// Pybind11 绑定
PYBIND11_MODULE(rigid_ransac, m) {
    m.def("svdICP", &svdICP, "SVD-based ICP for rigid transformation");
    m.def("rigid_ransac", &rigidRansac, 
          py::arg("points1"), py::arg("points2"), 
          py::arg("iters") = 1000, py::arg("threshold") = 0.5,
          "RANSAC-based rigid transformation estimation.");
}
