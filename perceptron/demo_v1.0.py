import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
author：gengo
感知机
借鉴：https://blog.csdn.net/red_stone1/article/details/80491895
收集数据：随机产生数据 总共产生100 X 2个数据
准备数据：使用DataFrame将随机产生的数据整理，最终产生的数据格式： x1 x2 y
分析数据：使用特征归一化优化 x1 x2
训练算法：随机生成w，使用矩阵乘法（np.dot()）完成计算
测试算法：
使用算法：
'''


# 随机产生数据（X,Y,Z 变量名无实际意义...）
def create_data():
    X = np.random.rand(50, 2) * 2
    X = np.hstack((np.zeros((50, 1)), X))
    X_data = pd.DataFrame(X)
    X_data.iloc[:, 0] = -1
    Y = np.random.uniform(2, 5, (50, 2))
    Y = np.hstack((np.ones((50, 1)), Y))
    Y_data = pd.DataFrame(Y)
    Z = X_data.append(Y_data)
    Z_data = pd.DataFrame(np.array(Z), columns=['y', 'x1', 'x2'])
    # 画出数据分布图
    plt.figure()
    plt.subplot(111)
    plt.axis([0, 5, 0, 5])
    plt.scatter(Z.iloc[0:50, 1], Z.iloc[0:50, 2], color='red', marker='x', s=10, alpha=0.75, label='positive')
    plt.scatter(Z.iloc[50:100, 1], Z.iloc[50:100, 2], color='blue', marker='o', s=10, alpha=0.75, label='negative')
    plt.title('original data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.show()
    return Z_data.loc[:, ['x1', 'x2', 'y']]


# 特征归一化
def data_optimize():
    Z = create_data()
    # 均值
    u = np.mean(Z.iloc[:, 0:2], axis=0)
    # 标准差
    v = np.std(Z.iloc[:, 0:2], axis=0)
    # X = (X - u)/v
    Z.iloc[:, 0:2] = (Z.iloc[:, 0:2] - u) / v
    # 画出特征值优化后的图
    plt.figure()
    plt.subplot(111)
    plt.scatter(Z.iloc[0:50, 0], Z.iloc[0:50, 1], color='red', marker='x', s=10, alpha=0.75, label='positive')
    plt.scatter(Z.iloc[50:100, 0], Z.iloc[50:100, 1], color='blue', marker='o', s=10, alpha=0.75, label='negative')
    plt.title('Normalization data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.show()
    return Z


# 直线初始化
def line_initialize():
    Z = data_optimize()
    # 增加x0列 w^('^'在w正上方)=(w0,w1,w2)T(转置)   Z^-> x0 x1 x2 y    w0视为b（偏置bias）
    Z = np.hstack((np.ones((np.shape(Z)[0], 1)), Z))
    Z = pd.DataFrame(Z, columns=['x0', 'x1', 'x2', 'y'])
    w = np.random.randn(3, 1)
    # WX + b = 0 ---> w0*x0+w1*x1+w2*x2=0 两点确定一条直线
    # 第一点
    x1 = -1
    y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
    # 第二点
    x2 = 1
    y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
    # 画出初始化的直线
    plt.figure()
    plt.subplot(111)
    plt.scatter(Z.iloc[0:50, 1], Z.iloc[0:50, 2], color='red', marker='x', s=10, alpha=0.75, label='positive')
    plt.scatter(Z.iloc[50:100, 1], Z.iloc[50:100, 2], color='blue', marker='o', s=10, alpha=0.75, label='negative')
    plt.plot([x1, x2], [y1, y2], 'r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    return Z, w


# 训练感知机
def perceptron():
    # 循环版本 （有时在100次循环之后依然没有效果）
    Z, w = line_initialize()
    # for i in np.arange(100):
    #     if (w[0]*Z.iloc[i,0]+w[1]*Z.iloc[i,1]+w[2]*Z.iloc[i,2])*Z.iloc[i,3] <= 0:
    #         w += Z.iloc[i,3] * np.array(Z.iloc[i,:3]).reshape((3,1))
    # return Z,w
    # 矩阵乘法 （能很好的进行分类）
    # 思路：将Z与w进行矩阵乘法，挑选出结果值为负数的行，然后与y做对比，即可知道该点分类正确与否，然后修正第一个错误的点
    for i in np.arange(100):
        s = np.dot(Z.iloc[:, :3], w)
        y = np.ones_like(Z.iloc[:, 3])
        loc_n = np.where(s < 0)[0]
        y[loc_n] = -1
        num_fault = len(np.where(Z.iloc[:, 3] != y)[0])
        print('第%2d次更新，分类错误的点的个数：%2d' % (i, num_fault))
        if num_fault == 0:
            break
        else:
            t = np.where(Z.iloc[:, 3] != y)[0][0]
            w += Z.iloc[t, 3] * np.array(Z.iloc[t, :3]).reshape((3, 1))
    return Z, w


# 训练结果展示
def res_display():
    Z, w = perceptron()
    x1 = -1
    y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
    x2 = 1
    y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
    # 结果图展示
    plt.figure()
    plt.subplot(111)
    plt.scatter(Z.iloc[0:50, 1], Z.iloc[0:50, 2], color='red', marker='x', s=10, alpha=0.75, label='positive')
    plt.scatter(Z.iloc[50:100, 1], Z.iloc[50:100, 2], color='blue', marker='o', s=10, alpha=0.75, label='negative')
    plt.plot([x1, x2], [y1, y2], 'r')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    res_display()
