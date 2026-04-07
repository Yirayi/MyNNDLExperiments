"""
验证sklearn中的rbf核的gamma=1.0 / 2*sigma^2
"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

x1 = np.array([[1, 2, 1]])
x2 = np.array([[0, 4, -1]])

gamma = 0.5
result = rbf_kernel(x1, x2, gamma=gamma)   # sklearn 计算结果
print(result)
# result ≈ 0.011109

# 手动验证：
sq_dist = 9   # ||x1-x2||^2
manual = np.exp(-gamma * sq_dist)          # exp(-0.5 * 9) ≈ 0.011109
print(manual)
# 两者相等 → 确认公式是 exp(-gamma * ||x-y||^2)
