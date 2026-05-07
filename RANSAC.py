import numpy as np 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "RANSACCPP"))
import rigid_ransac


def svdICP(src,dst):
    src = np.array(src).T
    dst = np.array(dst).T
    
    mean_src = np.mean(np.array(src),axis=1,keepdims=True)
    mean_dst = np.mean(np.array(dst),axis=1,keepdims=True)

    src_norm = src-mean_src
    dst_norm = dst-mean_dst

    mat_s = src_norm.dot(dst_norm.T)
    u, sigma, v_t = np.linalg.svd(mat_s)
    temp = u.dot(v_t)

    det = np.linalg.det(temp)
    s = np.array([[1,0],[0,det]])

    mat_r = v_t.T.dot(s).dot(u.T)

    translation = mean_dst.T - mean_src.T.dot(mat_r.T)
    return np.hstack((mat_r, translation.reshape(-1,1)))


def rigidRansac(points1, points2,iters = 1000):
    points1 = points1[:,[1,0]]
    points2 = points2[:,[1,0]]
    max_cs_num = 0
    mask = np.zeros((points1.shape[0],1))
    mat = np.zeros((2,3))
    for i in range(iters):
        idx1 = np.random.randint(points1.shape[0])
        idx2 = np.random.randint(points1.shape[0])
        # idx3 = np.random.randint(points1.shape[0])
        x = points1[[idx1,idx2], :]
        y = points2[[idx1,idx2], :]
        
        rot_mat = svdICP(x,y) 
        y_hat = points1.dot(rot_mat[:2,:2].T)+rot_mat[:2,2]
        err = np.abs(y_hat - points2)
        err = np.sqrt(np.sum(err**2, axis=1))

        consensus_num = np.sum(err < 0.5)
        if consensus_num > max_cs_num:
            max_cs_num = consensus_num
            mask = err < 0.5
            mat = rot_mat

    points1_c = points1[mask]
    points2_c = points2[mask]

    mat = svdICP(points1_c, points2_c)
    
    return mat , mask, max_cs_num






def generate_test_data(num_points=100, noise_level=0.05, rotation_angle=np.pi/4, translation=np.array([2, 3])):
    # 随机生成原始点集
    points1 = np.random.rand(num_points, 2) * 10
    
    # 生成旋转矩阵
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ])
    
    # 对点集进行旋转和平移
    points2 = points1.dot(rotation_matrix.T) + translation
    
    # 添加噪声
    noise = np.random.randn(*points2.shape) * noise_level
    points2_noisy = points2 + noise
    
    return points1, points2_noisy

def compare_ransac_results(points1, points2):
    # 调用 Python 版本的 RANSAC
    mat_py, mask_py, max_cs_num_py = rigidRansac(points1, points2)
    print("Python RANSAC:")
    print("Transformation Matrix:\n", mat_py)
    print("Consensus Set Size:", max_cs_num_py)
    
    # 调用 C++ 版本的 RANSAC
    mat_cpp,mask_pc, inlier_count_cpp = rigid_ransac.rigid_ransac(points1, points2)

    mat_cpp[:2,:2] = np.linalg.inv(mat_cpp[:2,:2])
    mat_cpp[:2, -1] = mat_cpp[:2, -1][::-1]
    mat_cpp = mat_cpp[:2,:]
    print("\nC++ RANSAC:")
    print("Transformation Matrix:\n", mat_cpp)
    print("Consensus Set Size:", inlier_count_cpp)
    
    # 对比两个结果
    print("\nComparison:")
    print("Transformation Difference:\n", np.abs(mat_py - mat_cpp))
    print("Consensus Set Size Difference:", abs(max_cs_num_py - inlier_count_cpp))

if __name__ == "__main__":
    points1, points2 = generate_test_data()
    compare_ransac_results(points1, points2)
