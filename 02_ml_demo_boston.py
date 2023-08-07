import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=13, out_features=6),  # 第一个全连接 - 隐层
            nn.Sigmoid(),
            nn.Linear(in_features=6, out_features=3),  # 第二个全连接 - 隐层
            nn.Sigmoid(),
            nn.Linear(in_features=3, out_features=1)  # 第三个全连接 - 输出层
        )

    def forward(self, x):
        """
        定义Network这个网络的前向执行过程，forward方法的入参可以是任意的
        :param x: 起始就是网络/模型的原始输入，此时是boston房价预测的特征属性数据，shape为[N,13]表示N个样本，13个特征属性
        :return: 模型最终预测输出值，此时是boston房价预测价格，shape为[N,1]表示N个样本，1表示一个输出值
        """
        return self.model(x)


class NumpyDataset(Dataset):

    def __init__(self, x, y):
        super(NumpyDataset, self).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        # 返回index对应的样本数据
        return self.x[index], self.y[index]

    def __len__(self):
        # 返回当前数据集的样本条数
        return len(self.x)


def fetch_dataloader(batch_size):
    # 1. 加载数据 + 数据特征工程
    X, Y = datasets.load_boston(return_X_y=True)
    Y = Y.reshape(-1, 1).astype('float32')
    X = X.astype('float32')
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=24)
    x_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)
    print(f"训练数据shape:{train_x.shape} - {train_y.shape}")
    print(f"训练数据shape:{test_x.shape} - {test_y.shape}")

    # 2. 构建Dataset对象
    train_dataset = NumpyDataset(x=train_x, y=train_y)
    test_dataset = NumpyDataset(x=test_x, y=test_y)

    # 3. 构建数据遍历器
    # 将dataset里面的数据一条一条的拿出来，然后合并到一起形成一个批次的数据集，并返回
    train_dataloader = DataLoader(
        dataset=train_dataset,  # 给定数据集对象，要求必须有__getitem__方法
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 在从dataset中提取数据的时候，是否需要打乱顺序
        num_workers=0,  # 数据加载形成batch的过程是否需要多线程，0表示直接在当前主线程中执行
        collate_fn=None,  # 给定如何将n条数据合并成批次数据返回，默认情况不用调整
        prefetch_factor=2  # 当num_workers为0的时候，必须为默认值；其它情况给定的是预加载的样本数目，一般情况设置为batch_size * num_workers
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
        collate_fn=None
    )
    return train_dataloader, test_dataloader, test_x, test_y


def save_model(path, net, epoch, train_batch, test_batch):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'net': net,
        'epoch': epoch,
        'train_batch': train_batch,
        'test_batch': test_batch
    }, path)  # 底层调用python的pickle的API进行持久化操作


def training(restore_path=None):
    root_dir = './output/02'
    total_epoch = 10000000

    # 1. 数据加载
    train_dataloader, test_dataloader, test_x, test_y = fetch_dataloader(batch_size=32)

    # 2. 模型对象的构建
    net = Network()
    loss_fn = nn.MSELoss()
    opt = optim.SGD(params=net.parameters(), lr=0.005)
    # 模型恢复
    if (restore_path is not None) and os.path.exists(restore_path):
        original_net = torch.load(restore_path, map_location='cpu')
        net.load_state_dict(state_dict=original_net['net'].state_dict())  # 参数恢复
        train_batch = original_net['train_batch']
        test_batch = original_net['test_batch']
        start_epoch = original_net['epoch'] + 1
        total_epoch = total_epoch + start_epoch
    else:
        train_batch = 0
        test_batch = 0
        start_epoch = 0

    # 2.1. 运行状态可视化 -- 一般写在网络结构定义之后
    # 解决tensorflow框架来实现可视化，首先安装tensorflow框架：pip install tensorflow
    # 命令行执行如下命令: tensorboard --logdir xxxx
    writer = SummaryWriter(log_dir=f'{root_dir}/summary')
    writer.add_graph(net, input_to_model=torch.rand(1, 13))  # 添加执行图

    # 3. 模型训练
    # epoch: 将整个数据集从头到尾遍历一次叫一个epoch
    # batch: 一次前向 + 一次反向 就叫一个batch
    for epoch in range(start_epoch, total_epoch):
        net.train()
        train_loss = []
        for _x, _y in train_dataloader:
            # 1. 前向过程
            _pred_y = net(_x)  # 前向过程，获取推理预测值
            _loss = loss_fn(_pred_y, _y)  # 预测值和实际值计算损失

            # 2. 反向过程
            opt.zero_grad()  # 将所有待更新参数的梯度值重置为0
            _loss.backward()  # 反向传播求解梯度值
            opt.step()  # 参数更新
            train_batch += 1

            print(f"train epoch:{epoch}, batch:{train_batch}, loss:{_loss.item():.4f}")
            writer.add_scalar('train_batch_loss', _loss.item(), global_step=train_batch)
            train_loss.append(_loss.item())

        net.eval()  # 模型进入推理预测阶段
        test_loss = []
        with torch.no_grad():
            for _x, _y in test_dataloader:
                # 1. 前向过程
                _pred_y = net(_x)  # 前向过程，获取推理预测值
                _loss = loss_fn(_pred_y, _y)  # 预测值和实际值计算损失
                test_batch += 1

                print(f"test epoch:{epoch}, batch:{test_batch}, loss:{_loss.item():.4f}")
                writer.add_scalar('test_batch_loss', _loss.item(), global_step=test_batch)
                test_loss.append(_loss.item())

        # 可视化
        writer.add_histogram('w1', net.model[0].weight, global_step=epoch)
        writer.add_histogram('b1', net.model[0].bias, global_step=epoch)
        writer.add_histogram('w3', net.model[4].weight, global_step=epoch)
        writer.add_histogram('b3', net.model[4].bias, global_step=epoch)
        writer.add_scalars('loss', {'train': np.mean(train_loss), 'test': np.mean(test_loss)}, global_step=epoch)

        if epoch % 100 == 0:
            save_model(
                f'{root_dir}/model/net_{epoch}.pkl',
                net, epoch, train_batch, test_batch
            )

        # TODO: 自己加入提前结束训练的逻辑判断；eg: 连续五个epoch，测试数据集的测试效果没有提升，就终止模型 直接break

    writer.close()

    # 3. 模型测试数据评估
    net.eval()  # 模型进入推理预测阶段
    with torch.no_grad():
        predict_test_y = net(torch.from_numpy(test_x))
        predict_test_loss = loss_fn(predict_test_y, torch.from_numpy(test_y))
        print(np.hstack([predict_test_y.detach().numpy(), test_y]))
        print(predict_test_loss.item())

    # 4. 模型持久化
    save_model(
        f'{root_dir}/model/net_{epoch}.pkl',
        net, epoch, train_batch, test_batch
    )


def t1():
    net = Network()

    _x = torch.rand(8, 13)
    # net(_x) python语言知识 底层调用net对象的__call__方法  pytorch知识 pytorch中nn.Module对象的__call__方法指向了_call_impl方法，然后再_call_impl方法中会调用forward方法
    _r = net(_x)
    print(_r)


if __name__ == '__main__':
    training(restore_path='./output/02/net_100.pkl')
    # t1()
