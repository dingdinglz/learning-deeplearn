import torch
from matplotlib import pyplot as plt
import random

# 根据w ，b生成一组数据
def data_make(w,b,num):
    X = torch.normal(0,1,(num,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape(-1,1)

def choose_data(batch_size,dataSets,labels):
    chooseList = list(range(len(dataSets)))
    random.shuffle(chooseList)
    for i in range(0,len(dataSets),batch_size):
        getList = torch.tensor(chooseList[i:min(i+batch_size,len(dataSets))])
        yield dataSets[getList],labels[getList]

def loss(y,y_hat):
    return (y - y_hat.reshape(y.shape)) ** 2 / 2

def line(w,X,b):
    return torch.matmul(X,w) + b

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

if __name__ == '__main__':
    finalW = torch.tensor([2.0,-1.0])
    finalB = torch.tensor(3.0)
    dataSet,Labels = data_make(finalW,finalB,1000)
    plt.scatter(dataSet[:,1],Labels,1)
    plt.show()
    num_poll = 1000
    lr = 0.03
    batch_size = 3
    w = torch.zeros((2,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    for i in range(num_poll):
        for dataX,datay in choose_data(batch_size,dataSet,Labels):
            l = loss(line(w,dataX,b),datay)
            l.sum().backward()
            sgd([w,b],lr, batch_size)
        with torch.no_grad():
            train_l = loss(line(w,dataSet,b),Labels)
            print(f'Run {i+1} times : Now Loss : {float(train_l.mean()):f}')
    print(f'final w:{w},b:{b}')
    print('w误差：')
    for i in range(0,len(w)):
        print(f'w{i+1} : {(w[i][0] - finalW[i]):f}')
    print(f'b误差：{finalB - b.reshape(finalB.shape)}')