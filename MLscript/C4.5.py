from numpy import *
from scipy import *
from math import log
import pandas as pd
import random

random.seed ()

#储存决策树的类，由于针对了连续数据进行分类
#还不是很明白python中“树”这种数据结构的构建，主要是对于python中类似c的指针/变量等概念的不理解
class treeDict:
    def __init__ (self, flag, lable):
        self.name = lable #结点数据的类型名
        self.kind = flag  #表明该结点为连续（1）或离散（0）
        self.sep = 0      #连续数据中选取的分割点
        self.next = {}    #对于离散数据就是简单的每一个键值，连续则用0/1表示分割点左、右的取向

    # 决策树结构的输出，可以理解为是一种深度优先遍历
    def show (self): 
        print ("%s(%d):" %(self.name, len(self.next)), end="")
        if not len(self.next):
            return
        for i in self.next:
            print (i, end="\t")
            if type (self.next[i]) == treeDict:
                self.next[i].show ()
            else:
                print ("result:", self.next[i], end="\n")

    # 决策树分支的延申
    def grow (self, value, branch):
        self.next[value] = branch

#计算给定数据的香浓熵：
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  
    labelCounts = {}  #类别字典（类别的名称为键，该类别的个数为值）
    for featVec in dataSet:
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  #还没添加到字典里的类型
            labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1 
    shannonEnt = 0.0  
    for key in labelCounts:  #求出每种类型的熵
        prob = float(labelCounts[key])/numEntries  #每种类型个数占所有的比值
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt   #返回熵

#按照给定的特征划分数据集，对于连续数据与离散数据区分两个函数
def splitDataSet(dataSet, axis, value):
    retDataSet = []  
    for featVec in dataSet:  #按dataSet矩阵中的第axis列的值等于value的分数据集
        if featVec[axis] == value:      #值等于value的，每一行为新的列表（去除第axis个数据）
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  
            retDataSet.append(reducedFeatVec) 
    return retDataSet  #返回分类后的新矩阵

def splitDataSetX(dataSet, axis, value, flag):
    retDataSet = []  
    if not flag:
        for featVec in dataSet:  #按dataSet矩阵中的第axis列的值小于等于value的分数据集
            if featVec[axis] <= value:      #值小于等于value的，每一行为新的列表（去除第axis个数据）
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend (featVec[axis+1:])
                retDataSet.append (reducedFeatVec)
    else:
        for featVec in dataSet:  #按dataSet矩阵中的第axis列的值大于value的分数据集
            if featVec[axis] > value:      #值大于value的，每一行为新的列表（去除第axis个数据）
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet  #返回分类后的新矩阵

#选择最好的数据集划分方式
def chooseBestFeatureToSplit (dataSet):  
    numFeatures = len(dataSet[0])-1  #求属性的个数
    print (len(dataSet), "and", numFeatures, end="\t") # 输出了决策树的构建过程
    baseEntropy = calcShannonEnt (dataSet)
    bestInfoGain = 0.0  
    bestFeature = -1  
    result = []
    kind = -1
    for i in range (numFeatures):  #求所有属性的信息增益
        flag = 0
        featList = [example[i] for example in dataSet]  
        uniqueVals = set(featList)  #第i列属性的取值（不同值）数集合
        print (len(uniqueVals), end="\t")
        # 感觉这种对于连续/离散数据的区分方式不是太合适
        if len(uniqueVals)<=6 or type(featList[0]) == str:
            newEntropy = 0.0  
            splitInfo = 0.0 
            for value in uniqueVals:  #求第i列属性每个不同值的熵*他们的概率
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))  #求出该值在i列属性中的概率
                newEntropy += prob * calcShannonEnt(subDataSet)  #求i列属性各值对于的熵求和
                splitInfo -= prob * log(prob, 2) 
            if splitInfo == 0:
                continue
        else: # 对于连续数据的处理
            flag=1
            newEntropy = 1.0  
            splitInfo = 0.0 
            # 首先需要求出一个最佳分割点
            step = (max(uniqueVals) - min(uniqueVals))/10
            begin = min(uniqueVals)
            for j in range (9):
                begin += step
                EntropyTmp = 0.0
                splitInfoTmp = 0.0
                for k in range (2):
                    subDataSet = splitDataSetX (dataSet, i, begin, k)
                    prob = len(subDataSet)/float(len(dataSet)) 
                    EntropyTmp += prob * calcShannonEnt (subDataSet) 
                    splitInfoTmp -= prob * log(prob, 2)
                if EntropyTmp < newEntropy:
                    newEntropy=EntropyTmp
                    splitInfo=splitInfoTmp
                    sepTmp=begin
        infoGain = (baseEntropy - newEntropy) / splitInfo   #求出第i列属性的信息增益率
        #print(infoGain)
        if (infoGain > bestInfoGain):  #保存信息增益率最大的信息增益率值以及所在的下表（列值i）
            bestInfoGain = infoGain  
            bestFeature = i
            kind=flag
            if flag:
                sep=sepTmp
    print ()
    if kind == -1: 
        result.append (-1)
    # 这是运行当中遇到的一个意外，可能由于之前某次连续数据结点构建时分割点的选取不适当，导致本次划分选择时剩下的所有属性都是一样的但标签不一样
    # 当然也有可能是数据本身就不适用于决策树或错误样本等
    else: 
        result.append (bestFeature)
        if kind == 1:
            result.append (sep)
    return result 

#找出出现次数最多的分类名称
def majorityCnt (classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): 
            classCount[vote] = 0  
        classCount[vote] += 1  
    maxTimes = 0
    for i in classCount:
        if classCount[i] > maxTimes:
            major = i
            maxTimes = classCount[i]
    return major

#创建树
def createTree(dataSet, labels):  
    classList = [example[-1] for example in dataSet]     #创建需要创建树的训练数据的结果列表
    if classList.count (classList[0]) == len(classList):  #如果所有的训练数据都是属于一个类别，则返回该类别
        return classList[0]   
    if (len(dataSet[0]) == 1):  #训练数据只给出类别数据（没给任何属性值数据），返回出现次数最多的分类名称
        return majorityCnt (classList) 

    result = chooseBestFeatureToSplit(dataSet)
    bestFeat = result[0]    #选择信息增益最大的属性进行分（返回值是属性类型列表的下标）
    if result[0] == -1:
        return majorityCnt (classList) 
    bestFeatLabel = labels[bestFeat]  #根据下表找属性名称当树的根节点
    del (labels[bestFeat])  #从属性列表中删掉已经被选出来当根节点的属性
    if len(result) == 1:
        myTree = treeDict (0, bestFeatLabel)
        featValues = [example[bestFeat] for example in dataSet]  #找出该属性所有训练数据的值（创建列表）
        uniqueVals = set(featValues)  #求出该属性的所有值得集合（集合的元素不能重复）
        for value in uniqueVals:  #根据该属性的值求树的各个分支
            subLabels = labels[:]  
            myTree.grow (value, createTree(splitDataSet (dataSet, bestFeat, value), subLabels))  # 根据各个分支递归创建树
    else: # 连续数据的分支处理
        myTree = treeDict (1, bestFeatLabel)
        myTree.sep = result[1]
        subLabels = labels[:]
        myTree.grow (0, createTree(splitDataSetX (dataSet, bestFeat, result[1], 0), subLabels))  # 左递归创建树
        subLabels = labels[:]
        myTree.grow (1, createTree(splitDataSetX (dataSet, bestFeat, result[1], 1), subLabels))  # 右分支递归创建树
    return myTree  #生成的树

#实用决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.name
    #print (firstStr, end="\t")
    featIndex = featLabels.index(firstStr)
    if not inputTree.kind: 
        flag = 0
        for key in inputTree.next:  
            if testVec[featIndex] == key:  
                flag = 1
                if type(inputTree.next[key]) == treeDict:  
                    classLabel = classify(inputTree.next[key], featLabels, testVec)  
                else: 
                    classLabel = inputTree.next[key]
        if not flag: # 这也是预测过程中遇到的一个问题，在一个下层结点中发现键值中没有符合预测目标的，于是选择最后一个键值走下去
            if type(inputTree.next[key]) == treeDict:  
                classLabel = classify(inputTree.next[key], featLabels, testVec)  
            else: 
                classLabel = inputTree.next[key]
    else:
        key = testVec[featIndex] > inputTree.sep
        if type(inputTree.next[key]) == treeDict:  
            classLabel = classify(inputTree.next[key], featLabels, testVec)  
        else: 
            classLabel = inputTree.next[key]
    return classLabel  

#读取数据文档中的训练数据（生成二维列表）
def getData():
    raw = pd.read_csv ('bank.csv', sep=";")
    raw_data = raw.values
    random.shuffle (raw_data)
    dataSet = []
    for i in raw_data[:]:
        tmp = list (i)
        dataSet.append (tmp)
    labels = list (raw.columns)
    return dataSet, labels

def main ():
    myDat, labels = getData()  
    train = myDat[:35000]
    test = myDat[35001:]
    myTree = createTree (train, labels[:])
    #myTree.show ()
    right=0
    wrong=0
    for testData in test:
        dic = classify(myTree, labels, testData)
        if dic == testData[-1]:
            right+=1
        else:
            wrong+=1
        print (dic, "\t", testData[-1])
    print ("%6f" %(right/(right+wrong)))

if __name__ == '__main__':
    main()
