from __future__ import division
import math
import random
import numpy as np
import scipy.io as scio

random.seed(0)

# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b - a) * random.random() + a

# 生成大小 I*J 的矩阵，默认零矩阵
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m

# 函数 sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# 函数 sigmoid 的导数
def dsigmoid(x):
    return x * (1 - x)

class NN:
    def __init__(self, ni, nh1, nh2, no):
        self.ni = ni + 1
        self.nh1 = nh1 + 1
        self.nh2 = nh2 + 1
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah1 = [1.0] * self.nh1
        self.ah2 = [1.0] * self.nh2
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni, self.nh1)
        self.wh = makeMatrix(self.nh1, self.nh2)
        self.wo = makeMatrix(self.nh2, self.no)

        for i in range(self.ni):
            for j in range(self.nh1):
                self.wi[i][j] = rand(-0.2, 0.2)
        for i in range(self.nh1):
            for j in range(self.nh2):
                self.wh[i][j] = rand(-0.2, 0.2)
        for i in range(self.nh2):
            for j in range(self.no):
                self.wo[i][j] = rand(-2, 2)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('与输入层节点数不符！')
        # 激活输入层
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]
        # 激活隐藏层
        for j in range(self.nh1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah1[j] = sigmoid(sum)

        for j in range(self.nh2):
            sum = 0.0
            for i in range(self.nh1):
                sum = sum + self.ah1[i] * self.wh[i][j]
            self.ah2[j] = sigmoid(sum)
        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh2):
                sum = sum + self.ah2[j] * self.wo[j][k]
            self.ao[k] = sum

        return self.ao[:]

    def backPropagate(self, targets, lr):
        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            output_deltas[k] = targets[k] - self.ao[k]
        # 计算隐藏层的误差
        hidden_deltas2 = [0.0] * self.nh2
        for j in range(self.nh2):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas2[j] = dsigmoid(self.ah2[j]) * error

        hidden_deltas1 = [0.0] * self.nh1
        for j in range(self.nh1):
            error = 0.0
            for k in range(self.nh2):
                error = error + hidden_deltas2[k] * self.wh[j][k]
            hidden_deltas1[j] = dsigmoid(self.ah1[j]) * error
        # 更新输出层权重
        for j in range(self.nh2):
            for k in range(self.no):
                change = output_deltas[k] * self.ah2[j]
                self.wo[j][k] = self.wo[j][k] + lr * change

        for j in range(self.nh1):
            for i in range(self.nh2):
                change = hidden_deltas2[i] * self.ah1[j]
                self.wh[j][k] = self.wh[j][i] + lr * change
        # 更新输入层权重
        for i in range(self.ni):
            for j in range(self.nh1):
                change = hidden_deltas1[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + lr * change
        # 计算误差
        error = 0.0
        error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            target = p[1]
            result = self.update(p[0])
            print(p[0], ':', target, '->', result)
        #accuracy = float (count / len(patterns))
        #print('accuracy: %-.9f' % accuracy)

    def weights(self):
        print('输入层1权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输入层2权重:')
        for i in range(self.nh1):
            print(self.wh[i])
        print()
        print('输出层权重:')
        print()
        for j in range(self.nh2):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, lr=0.01):
        # lr: 学习速率(learning rate)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update (inputs)
                error = error + self.backPropagate(targets, lr)
            if i % 100 == 0:
                print('error: %-.9f' % error)

def main():
    data = []
    offline_data = scio.loadmat('offline_data_random.mat')
    online_data = scio.loadmat('online_data.mat')
    offline_location, offline_rss = offline_data['offline_location'], offline_data['offline_rss']
    trace, rss = online_data['trace'][0:1000, :], online_data['rss'][0:1000, :]

    for i in range (6):
        column = [rec[i] for rec in offline_rss]
        Max = max (column)
        Min = min (column)
        for j in range (len (offline_rss)):
            offline_rss[j][i] = (offline_rss[j][i] - Min) / (Max - Min)
    column = [rec[0] for rec in offline_location]
    Max = max (column)
    Min = min (column)
    for j in range (len (offline_location)):
        offline_location[j][0] = offline_location[j][0] / 20
    column = [rec[1] for rec in offline_location]
    Max = max (column)
    Min = min (column)
    for j in range (len (offline_location)):
        offline_location[j][1] = offline_location[j][1] / 15
    
    for i in range (6):
        column = [rec[i] for rec in rss]
        Max = max (column)
        Min = min (column)
        for j in range (len (rss)):
            rss[j][i] = (rss[j][i] - Min) / (Max - Min)
    column = [rec[0] for rec in trace]
    Max = max (column)
    Min = min (column)
    for j in range (len (trace)):
        trace[j][0] = trace[j][0] / 20
    column = [rec[1] for rec in trace]
    Max = max (column)
    Min = min (column)
    for j in range (len (trace)):
        trace[j][1] = trace[j][1] / 15
    
    for i in range (len (offline_rss)):
        ele = []
        ele.append (offline_rss[i])
        ele.append (offline_location[i])
        data.append (ele)
    print (data)
    random.shuffle (data)
    training = data [0:15000]
    test = data [15000:]
    nn = NN (6, 100, 100, 2)
    nn.train (training, iterations=10000)
    nn.test (test)

if __name__ == '__main__':
    main()
