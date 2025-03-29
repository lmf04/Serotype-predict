import tensorflow as tf
import numpy as np

# 创建一个形状为 (2, 2) 的矩阵，每个元素初始化为0
large_matrix = np.zeros((3, 3))
 
# 创建一个2x2的NumPy数组，作为要插入的子矩阵
small_matrix = np.array([[1, 2], [3, 4]])
 
# 指定要插入子矩阵的位置（例如，从large_matrix的(1, 1)位置开始）
row_start, col_start = 1, 1
 
# 检查是否可以插入（不越界）
if row_start + small_matrix.shape[0] <= large_matrix.shape[0] and col_start + small_matrix.shape[1] <= large_matrix.shape[1]:
    large_matrix[row_start:row_start + small_matrix.shape[0], col_start:col_start + small_matrix.shape[1]] = small_matrix
 
print(large_matrix)