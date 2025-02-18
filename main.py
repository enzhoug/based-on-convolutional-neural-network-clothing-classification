import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import transforms
#加载数据集
def get_data(batch_size,resize=None):
    #使用插值法增加原始图片像素
    trans=[transforms.ToTensor()]#初始化包含 ToTensor 转换的列表
    if resize:
        trans.insert(0, transforms.Resize(resize))#在列表开头插入 Resize 转换
    trans=transforms.Compose(trans)#使用 Compose 将转换列表组合成一个转换序列
    #读取数据集
    minist_train = torchvision.datasets.FashionMNIST(root='C:\\Users\\15331\\data', train=True,
                                                     transform=trans)
    minist_test = torchvision.datasets.FashionMNIST(root='C:\\Users\\15331\\data', train=False,
                                                    transform=trans)
    # print(len(minist_train),len(minist_test))
    # print(minist_test)
    train_loader = data.DataLoader(minist_train, batch_size=batch_size, shuffle=True,pin_memory=True)
    test_loader = data.DataLoader(minist_test, batch_size=batch_size, shuffle=False,pin_memory=True )
    return train_loader, test_loader
#搭建卷积神经网络
def LeNet():
    net = nn.Sequential(
        nn.Conv2d(1, 6, (5, 5), 1, 2), nn.Sigmoid(),
        nn.AvgPool2d((2, 2), 2),
        nn.Conv2d(6, 16, (5, 5)), nn.Sigmoid(),
        nn.AvgPool2d((2, 2), 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    return net
def AlexNet():
    net = nn.Sequential(
        nn.Conv2d(1,96,(11,11),4,1),nn.ReLU(),
        nn.MaxPool2d((3,3),2),
        nn.Conv2d(96,256,(5,5),1,2),nn.ReLU(),
        nn.MaxPool2d((3,3),2),
        nn.Conv2d(256,384,(3,3),1,1),nn.ReLU(),
        nn.Conv2d(384,384,(3,3),1,1),nn.ReLU(),
        nn.Conv2d(384,256,(3,3),1,1),nn.ReLU(),
        nn.MaxPool2d((3,3),2),
        nn.Flatten(),
        nn.Linear(6400,4096),nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096,10)
    )
    return net
#x=torch.ones((1,1,28,28))
#print(net(x))
#创建损失函数
loss=nn.CrossEntropyLoss()
loss.to(device=torch.device('cuda:0'))
#初始化weight,bias
def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
#训练
def trainer(lr,epochs,batch_size,resize=None,net=LeNet()):
    net.apply(init_weight)
    # 创建优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.to(torch.device('cuda:0'))#将net模型移动到GPU
    # net.cuda()
    y_l = []#以列表记录每轮最后一次训练数据loss
    y_lc = []#以列表记录每轮最后一次测试数据loss
    x_e = []#以列表记录epoch
    acc_c=[]#以列表记录每轮次测试集上的准确率
    train_loader, test_loader = get_data(batch_size,resize)#加载数据集
    for epoch in range(epochs):
        # 训练
        net.train()
        print("-----第{}轮训练-----".format(epoch+1))
        t = 0#记录每轮次训练数据次数
        tc = 0#记录每轮次测试数据次数
        x_e.append(epoch + 1)
        for x, y in train_loader:
            x, y = x.to(torch.device('cuda:0')), y.to(torch.device('cuda:0'))#将数据和标签移动到GPU
            optimizer.zero_grad()
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            t = t + 1
            if t % (math.ceil(60000/batch_size)) == 0:#仅打印每轮次最后一次的loss
                y_l.append(l.item())
                print("训练次数：{},loss：{}".format(t, l.item()))
        # 测试
        net.eval()
        l_sum = 0#每轮次测试损失的总值
        acc_total=0
        with torch.no_grad():#禁用梯度
            for x, y in test_loader:
                x, y = x.to(torch.device('cuda:0')), y.to(torch.device('cuda:0'))#将数据和标签移动到GPU
                y_hat = net(x)
                #print('------')
                #print(y_hat.item())
                #print(y_hat)
                #print(y)
                #print('------')
                acc=(y_hat.argmax(1)==y).sum()
                acc_total+=acc.item()#测试集上采用平均损失
                l = loss(y_hat, y)
                l_sum+=l.item()
                tc = tc + 1
                if tc % (math.ceil(10000/batch_size)) == 0:
                    y_lc.append(l.item())#.item()将单元素的张量（或数组）转换为基础的Python数值类型
                    print("测试次数：{},loss：{}".format(tc, l_sum/tc))
        acc_c.append(acc_total/10000)
        print(f"测试集上预测准确率{acc_total/10000}")
    return x_e, y_l, y_lc, acc_c
#作图
def depict(x_e, y_l, y_lc, acc_c,name):
    #第一张图记录训练和测试的loss
    plt.figure()
    l1 = plt.plot(x_e, y_l)  # x,y必须是列表
    l2 = plt.plot(x_e, y_lc)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.plot()返回的是一个列表，即使它只包含一个线条
    plt.legend([l1[0], l2[0]], ['train', 'test'], loc='best')  # 绘制注解
    plt.title(f'{name}:Training and Test Loss per Epoch')
    plt.show()
    #第二张图记录测试准确率
    plt.figure()
    l3 = plt.plot(x_e, acc_c)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(f'{name}:Test Accuracy per Epoch')
    plt.show()
#查看GPU是否可用
print(torch.cuda.is_available())
print(torch.cuda.device_count())
#调用上述封装好的函数
#使用LeNet训练
net1 = LeNet()
x_e,y_l,y_lc, acc_c = trainer(lr=0.4,epochs=10,batch_size=64,resize=None,net=net1)
depict(x_e,y_l,y_lc, acc_c,"LeNet")
#使用AlexNet训练
net2=AlexNet()
x_e, y_l, y_lc, acc_c= trainer(lr=0.01, epochs=10, batch_size=64,resize=224,net=net2)
depict(x_e, y_l, y_lc, acc_c,"AlexNet")

