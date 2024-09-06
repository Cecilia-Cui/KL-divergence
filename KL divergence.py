import numpy as np
import pickle

# 加载数据
with open('R_ET_1_ET_2_ET_3_0.pkl', 'rb') as file:
    data = pickle.load(file)

# 提取9x9的 Fisher 矩阵
fisher_matrix_1 = data[1]
fisher_matrix_2 = data[2][0]
fisher_matrix_3 = data[2][1]

# 动态正则化项函数
def get_dynamic_epsilon(matrix):
    cond_number = np.linalg.cond(matrix)
    eigvals = np.linalg.eigvals(matrix)
    eigval_std = np.std(eigvals)
    
    if cond_number > 1e20 or eigval_std > 1e2:
        return 1e-2
    elif cond_number > 1e16 or eigval_std > 1e1:
        return 1e-3
    elif cond_number > 1e12:
        return 1e-4
    else:
        return 1e-6

# 强化特征值裁剪，确保特征值在更严格的范围内
def ensure_positive_definite(matrix, matrix_name=""):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    
    # 打印特征值供调试
    print(f"{matrix_name} Eigenvalues before clipping: {eigvals}")
    
    # 将特征值限制在 [1e-2, 1e3] 之间，避免极端特征值
    eigvals_clipped = np.clip(eigvals, 1e-2, 1e3)
    
    # 打印裁剪后的特征值
    print(f"{matrix_name} Eigenvalues after clipping: {eigvals_clipped}")
    
    # 重构正定矩阵
    matrix_pos_def = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return matrix_pos_def

# 正则化矩阵
def regularize_matrix(matrix, matrix_name=""):
    epsilon = get_dynamic_epsilon(matrix)
    
    # 打印条件数供调试
    cond_number = np.linalg.cond(matrix)
    if cond_number > 1e12:
        print(f"Matrix condition number is high ({cond_number}), adding regularization with epsilon {epsilon}.")
    
    # 确保矩阵对称性
    matrix_sym = (matrix + matrix.T) / 2
    return ensure_positive_definite(matrix_sym + epsilon * np.eye(matrix.shape[0]), matrix_name)

# 计算协方差矩阵
def compute_covariance(fisher_matrix, matrix_name=""):
    reg_matrix = regularize_matrix(fisher_matrix, matrix_name)
    cov_matrix = np.linalg.pinv(reg_matrix)
    return cov_matrix

cov_matrix_1 = compute_covariance(fisher_matrix_1, "matrix_1")
cov_matrix_2 = compute_covariance(fisher_matrix_2, "matrix_2")
cov_matrix_3 = compute_covariance(fisher_matrix_3, "matrix_3")

# 打印验证
print("Shape of fisher_matrix_1:", fisher_matrix_1.shape)
print("Shape of cov_matrix_1:", cov_matrix_1.shape)

# 优化均值计算
mean1 = np.mean(cov_matrix_1, axis=0)
mean2 = np.mean(cov_matrix_2, axis=0)
mean3 = np.mean(cov_matrix_3, axis=0)

# 动态调节 KL 散度中的 Trace 和均值差项
def dynamic_clip(value, threshold=250):
    """动态限制数值的最大值"""
    return min(value, threshold)

def kl_divergence(cov1, mean1, cov2, mean2, matrix_name=""):
    """
    计算两个多元高斯分布之间的KL散度
    """
    # 正定化矩阵
    cov2_regularized = regularize_matrix(cov2, matrix_name)
    
    # 检查特征值
    eigvals_cov2 = np.linalg.eigvals(cov2_regularized)
    if np.any(eigvals_cov2 <= 0):
        print(f"Matrix {matrix_name} has non-positive eigenvalues, skipping KL computation.")
        return np.nan
    
    # 使用标准逆矩阵
    try:
        inv_cov2 = np.linalg.inv(cov2_regularized)
    except np.linalg.LinAlgError:
        print(f"Matrix {matrix_name} inversion failed, skipping KL computation.")
        return np.nan
    
    # 计算 Trace 项
    term1 = np.trace(inv_cov2 @ cov1)
    term1 = dynamic_clip(term1, 250)  # 限制 Trace 项
    print(f"{matrix_name} KL term1 (Trace): {term1}")
    
    # 计算行列式对数差
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2_regularized)
    
    if sign1 <= 0 or sign2 <= 0:
        print(f"Matrix {matrix_name} determinant is non-positive.")
        return np.nan
    
    term2 = logdet2 - logdet1
    print(f"{matrix_name} KL term2 (Log determinant difference): {term2}")
    
    # 计算均值差项
    term3 = (mean2 - mean1).T @ inv_cov2 @ (mean2 - mean1)
    term3 = dynamic_clip(term3, 250)  # 限制均值差项
    print(f"{matrix_name} KL term3 (Mean difference): {term3}")
    
    # 计算 KL 散度
    result = 0.5 * (term1 + term2 + term3 - cov1.shape[0])
    print(f"{matrix_name} KL result: {result}")
    
    return max(result, 0)

# 计算并输出每两个矩阵之间的KL散度
kl_12 = kl_divergence(cov_matrix_1, mean1, cov_matrix_2, mean2, "matrix_12")
kl_13 = kl_divergence(cov_matrix_1, mean1, cov_matrix_3, mean3, "matrix_13")
kl_23 = kl_divergence(cov_matrix_2, mean2, cov_matrix_3, mean3, "matrix_23")

print(f"KL Divergence between matrix 1 and matrix 2: {kl_12}")
print(f"KL Divergence between matrix 1 and matrix 3: {kl_13}")
print(f"KL Divergence between matrix 2 and matrix 3: {kl_23}")
