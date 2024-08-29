# try to make a svd for k = min(shape) instead of min(shape) - 1
import numpy as np
# from scipy.sparse.linalg import eigs
from scipy.sparse import lil_matrix
from scipy.linalg import eig

def svd(A, k=None, tol=1e-10):
    """手动实现奇异值分解，仅保留k个奇异值"""
    # 求A的转置A^T
    At = A.T

    # 计算A*A^T
    AAt = np.dot(At, A) # np.dot(A, At)
    #AtA = np.dot(At, A)
    # print(AAt.shape)

    # 计算A*A^T的特征值和特征向量
    # eigvals_AAt, eigvecs_AAt = np.linalg.eig(AAt)
    # eigvals_AtA, eigvecs_AtA = np.linalg.eig(AtA)
    eigvals_AAt, eigvecs_AAt = eig(AAt.toarray())
    #eigvals_AtA, eigvecs_AtA = eigs(AtA)

    # 特征值从大到小排序
    sorted_indices_AAt = np.argsort(eigvals_AAt)[::-1]
    #sorted_indices_AtA = np.argsort(eigvals_AtA)[::-1]

    # 提取非负特征值和对应的特征向量
    non_zero_eigvals_AAt = eigvals_AAt[sorted_indices_AAt]
    non_zero_eigvecs_AAt = eigvecs_AAt[:, sorted_indices_AAt]

    # 计算奇异值和奇异向量
    singular_values = np.sqrt(non_zero_eigvals_AAt)
    # singular_vectors = non_zero_eigvecs_AAt

    # 计算非零奇异值的数量和总和
    non_zero_count = np.count_nonzero(singular_values)
    sum_singular_values = np.sum(singular_values)
    # print(singular_values)
    '''
    # 仅保留k个奇异值
    if k is not None:
        if k > non_zero_count:
            k = non_zero_count
        singular_values = singular_values[:k]
        singular_vectors = singular_vectors[:, :k]
    # 计算A*A^T的奇异向量和奇异值
    V = singular_vectors
    S = np.diag(singular_values)
    U = np.dot(At, V) / singular_values
    '''
    return non_zero_count, sum_singular_values # U, S, V.T, non_zero_count, sum_singular_values
'''
# 示例矩阵
# 创建一个3x3的稀疏矩阵
A = lil_matrix((3, 2))

# 设置非零元素
A[0, 0] = 1
A[1, 1] = 2
A[2, 0] = 3

# 计算SVD
non_zero_count, sum_singular_values = svd(A)

# 输出全部奇异值
print("All singular values:", non_zero_count, sum_singular_values)
'''
