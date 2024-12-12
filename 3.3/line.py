import torch
from torch.utils import data
from torch import nn
from matplotlib import pyplot as plt

def synthetic_data(w,b,num_examples):
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape(-1,1)

def load_dataArray(dataarray,batch_size,is_train = True):
    dataset = data.TensorDataset(*dataarray)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    plt.scatter(features[:,1],labels,1)
    plt.show()
    batch_size = 10
    data_itor = load_dataArray((features,labels) , batch_size)
    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
    loss = nn.HuberLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    epoch_nums = 10
    for epoch_num in range(epoch_nums):
        for X, y in data_itor:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        with torch.no_grad():
            l = loss(net(features),labels)
            print(f'epoch: {epoch_num + 1} Loss: {l:f}')
    print(net[0].weight.data)
    print(net[0].bias.data)