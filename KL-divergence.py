import numpy as np
import pickle

# 加载数据
with open('R_ET_1_ET_2_ET_3_0.pkl', 'rb') as file:
    data = pickle.load(file)

# 提取协方差矩阵
cov_matrix_1 = data[1]  # 直接使用 data[1] 作为第一个协方差矩阵
cov_matrix_2 = data[2][0]  # data[2] 是一个包含两个 9x9 矩阵的三维数组
cov_matrix_3 = data[2][1]  # 提取第二个 9x9 矩阵

# 打印每个矩阵的形状以验证
print("Shape of cov_matrix_1:", cov_matrix_1.shape)
print("Shape of cov_matrix_2:", cov_matrix_2.shape)
print("Shape of cov_matrix_3:", cov_matrix_3.shape)

def kl_divergence(cov1, cov2):
    """
    计算两个多元高斯分布之间的KL散度
    """
    # 正则化防止不可逆情况
    cov2 += np.eye(cov2.shape[0]) * 1e-10
    
    inv_cov2 = np.linalg.inv(cov2)
    term1 = np.trace(inv_cov2 @ cov1)
    term2 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    k = cov1.shape[0]  # 矩阵的维度
    return 0.5 * (term1 + term2 - k)

# 计算并输出每两个矩阵之间的KL散度
kl_12 = kl_divergence(cov_matrix_1, cov_matrix_2)
kl_13 = kl_divergence(cov_matrix_1, cov_matrix_3)
kl_23 = kl_divergence(cov_matrix_2, cov_matrix_3)

print(f"KL Divergence between matrix 1 and matrix 2: {kl_12}")
print(f"KL Divergence between matrix 1 and matrix 3: {kl_13}")
print(f"KL Divergence between matrix 2 and matrix 3: {kl_23}")
