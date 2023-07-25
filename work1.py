import numpy as np

def softmax1(x):
    # 计算指数
    e_x = np.exp(x - np.max(x))
    # 计算概率
    return e_x / np.sum(e_x)

# 输入一个3维向量
x = np.array([1, 2, 3])
# 使用softmax函数得到输出向量
output = softmax1(x)

print(x.shape)
print(output.shape)
print(x)
print(output)






