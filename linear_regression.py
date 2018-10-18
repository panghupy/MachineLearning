import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    '''线性回归'''

    def __init__(self):
        # 初始化一些数据
        self.alpha = 0.01  # 学习率
        self.min_change = 1000  # 阈值

    # 　加载数据
    def load_data(self, filename):
        data = np.loadtxt(filename, delimiter=',')
        # 　样本数量
        m = data.shape[0]
        # 　特征数量
        n = data[:, 0:-1].shape[1]
        X0 = np.ones((m, 1))
        X1 = np.hstack((X0, data[:, :-1]))
        # 　特征缩放
        X, mu, sigma = self.featureScale(X1)
        Y = data[:, -1].reshape(m, 1)
        return X, Y, m, n, mu, sigma

    # 代价函数
    def computeCost(self, X, selfY, theta):
        error = np.dot(X, theta) - Y
        J = np.dot(error.T, error) / (2 * m)
        return J

    # 特征缩放,X:n+1维矩阵
    def featureScale(self, X):
        data = X[:, 1:]
        mu = np.mean(data, axis=0)  # 注：axis=0,对列进行计算，与octave中相似
        sigma = np.std(data, axis=0)
        data_scale = (data - mu) / sigma
        # 将X0拼接上
        data = np.hstack((X[:, 0].reshape(len(X), 1), data_scale))
        return data, mu, sigma

    # 计算梯度
    def gradientDescent(self, X, Y, theta):
        error = np.dot(X, theta) - Y
        gradient = np.dot(X.T, error) / m
        return gradient

    # 使用梯度下降迭代算法
    def train(self, X, Y):
        theta = np.ones((n + 1, 1))  # 初始化Theta
        gradient = self.gradientDescent(X, Y, theta)
        iter_num = 0
        # 代价函数变化记录
        J_history_list = []
        while not np.all(np.absolute(gradient) <= self.min_change):
            theta = theta - self.alpha * gradient
            gradient = self.gradientDescent(X, Y, theta)
            iter_num += 1
            J_history_list.append(self.computeCost(X, Y, theta)[0][0])
        return theta, iter_num, J_history_list


if __name__ == '__main__':
    print('Running gradient descent ...\n')
    L = LinearRegression()
    X, Y, m, n, mu, sigma = L.load_data('ex1data2.txt')
    theta, iter_num, J_history_list = L.train(X, Y)
    # 画出代价函数与的迭代次数的曲线图
    print('Theta computed from gradient descent:', theta)
    print('Number of iterations:\n', iter_num)
    print('Cost J:\n', L.computeCost(X, Y, theta))
    plt.plot(np.arange(0, iter_num, 1), J_history_list, '-r')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()
