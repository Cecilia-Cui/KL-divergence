import os
import numpy as np
import pickle
import csv

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
    eigvals_clipped = np.clip(eigvals, 1e-2, 1e3)
    matrix_pos_def = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return matrix_pos_def

# 正则化矩阵
def regularize_matrix(matrix, matrix_name=""):
    epsilon = get_dynamic_epsilon(matrix)
    matrix_sym = (matrix + matrix.T) / 2
    return ensure_positive_definite(matrix_sym + epsilon * np.eye(matrix.shape[0]), matrix_name)

# 计算协方差矩阵
def compute_covariance(fisher_matrix, matrix_name=""):
    reg_matrix = regularize_matrix(fisher_matrix, matrix_name)
    cov_matrix = np.linalg.pinv(reg_matrix)
    return cov_matrix

# 动态调节 KL 散度中的 Trace 和均值差项
def dynamic_clip(value, threshold=250):
    return min(value, threshold)

# 计算 KL 散度
def kl_divergence(cov1, mean1, cov2, mean2, matrix_name=""):
    cov2_regularized = regularize_matrix(cov2, matrix_name)
    eigvals_cov2 = np.linalg.eigvals(cov2_regularized)
    if np.any(eigvals_cov2 <= 0):
        return np.nan
    inv_cov2 = np.linalg.inv(cov2_regularized)
    term1 = dynamic_clip(np.trace(inv_cov2 @ cov1), 250)
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2_regularized)
    if sign1 <= 0 or sign2 <= 0:
        return np.nan
    term2 = logdet2 - logdet1
    term3 = dynamic_clip((mean2 - mean1).T @ inv_cov2 @ (mean2 - mean1), 250)
    result = 0.5 * (term1 + term2 + term3 - cov1.shape[0])
    return max(result, 0)

# 处理所有文件夹并计算KL散度
def process_all_folders(base_dir):
    results = []
    
    for i in range(200):  # 将范围限定在 0 到 199
        folder_path = os.path.join(base_dir, f'output_{i}', 'R_ET_1_ET_2_ET_3')
        pkl_file_path = os.path.join(folder_path, 'R_ET_1_ET_2_ET_3_0.pkl')
        
        if not os.path.exists(pkl_file_path):
            print(f"File {pkl_file_path} not found, skipping.")
            continue
        
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)

        fisher_matrix_1 = data[1]
        fisher_matrix_2 = data[2][0]
        fisher_matrix_3 = data[2][1]

        cov_matrix_1 = compute_covariance(fisher_matrix_1, "matrix_1")
        cov_matrix_2 = compute_covariance(fisher_matrix_2, "matrix_2")
        cov_matrix_3 = compute_covariance(fisher_matrix_3, "matrix_3")

        mean1 = np.mean(cov_matrix_1, axis=0)
        mean2 = np.mean(cov_matrix_2, axis=0)
        mean3 = np.mean(cov_matrix_3, axis=0)

        kl_12 = kl_divergence(cov_matrix_1, mean1, cov_matrix_2, mean2, "matrix_12")
        kl_13 = kl_divergence(cov_matrix_1, mean1, cov_matrix_3, mean3, "matrix_13")
        kl_23 = kl_divergence(cov_matrix_2, mean2, cov_matrix_3, mean3, "matrix_23")

        results.append({
            'folder': f'output_{i}',
            'kl_12': kl_12,
            'kl_13': kl_13,
            'kl_23': kl_23
        })
        
        print(f"Processed output_{i}: KL_12={kl_12}, KL_13={kl_13}, KL_23={kl_23}")
    
    return results

# 保存结果到CSV文件
def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['folder', 'kl_12', 'kl_13', 'kl_23'])
        writer.writeheader()
        writer.writerows(results)

# 主程序
if __name__ == "__main__":
    base_dir = "/Users/ciel/ET/KL_test_200Mpc"
    output_file = "kl_divergence_results_0_to_199.csv"
    
    results = process_all_folders(base_dir)
    save_results_to_csv(results, output_file)

    print(f"KL divergence results saved to {output_file}")
