from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
import torch.nn as nn
from torch import optim
from torch.onnx import TrainingMode
from torch.utils.tensorboard import SummaryWriter

from modules.metrics import Accuracy
from modules.numpy_dateset import build_dataloader


class IrisNetwork(nn.Module):
    def __init__(self):
        super(IrisNetwork, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_features=4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3)
        )

    def forward(self, x):
        """
        鸢尾花数据分类前向执行过程
        :param x: [N,4] N表示N个样本，4表示每个样本有4个特征属性
        :return: [N,3] scores N表示N个样本，3表示3个类别，每个样本属于每个类别的置信度值
        """
        return self.classify(x)


def save(obj, path):
    # 底层实际上就是Python的pickle的二进制保存
    torch.save(obj, path)


def load(path, net):
    print(f"模型恢复:{path}")
    ss_model = torch.load(path, map_location='cpu')
    net.load_state_dict(state_dict=ss_model['net'].state_dict(), strict=True)
    start_epoch = ss_model['epoch'] + 1
    best_acc = ss_model['acc']
    train_batch = ss_model['train_batch']
    test_batch = ss_model['test_batch']
    return start_epoch, best_acc, train_batch, test_batch


def training():
    # now = datetime.now().strftime("%y%m%d%H%M%S")
    now = '230720204537'
    root_dir = Path(f'./output/01/{now}')
    summary_dir = root_dir / 'summary'
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True)
    checkout_dir = root_dir / 'model'
    if not checkout_dir.exists():
        checkout_dir.mkdir(parents=True)
    last_path = checkout_dir / 'last.pkl'
    best_path = checkout_dir / 'best.pkl'
    total_epoch = 100
    summary_interval_batch = 2
    save_interval_epoch = 2
    start_epoch = 0
    best_acc = -1.0
    train_batch = 0
    test_batch = 0

    # 1. 定义数据加载器
    X, Y = load_iris(return_X_y=True)
    X = X.astype('float32')
    Y = Y.astype('int64')
    train_dataloader, test_dataloader, test_x, test_y = build_dataloader(X, Y, test_size=0.1, batch_size=32)

    # 2. 定义模型
    net = IrisNetwork()
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.005)

    # 3. 模型恢复
    if best_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(best_path, net)
    elif last_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(last_path, net)

    # 4. 定义可视化输出
    writer = SummaryWriter(log_dir=summary_dir)
    writer.add_graph(net, torch.rand(3, 4))

    # 5. 遍历训练模型
    for epoch in range(start_epoch, total_epoch + start_epoch):
        # 5.1 训练
        net.train()
        train_losses = []
        train_true, train_total = 0, 0
        for x, y in train_dataloader:
            # 前向过程
            scores = net(x)  # [N,3] 得到的是每个样本属于各个类别的置信度
            loss = loss_fn(scores, y)
            n, acc = acc_fn(scores, y)
            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss = loss.item()
            acc = acc.item()
            train_total += n
            train_true += n * acc
            if train_batch % summary_interval_batch == 0:
                print(f"epoch:{epoch}, train batch:{train_batch}, loss:{loss:.3f}, acc:{acc:.3f}")
                writer.add_scalar('train_loss', loss, global_step=train_batch)
                writer.add_scalar('train_acc', acc, global_step=train_batch)
            train_batch += 1
            train_losses.append(loss)

        # 5.2 评估
        net.eval()
        test_losses = []
        test_true, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_dataloader:
                # 前向过程
                scores = net(x)
                loss = loss_fn(scores, y)
                n, acc = acc_fn(scores, y)

                loss = loss.item()
                acc = acc.item()
                test_total += n
                test_true += n * acc
                print(f"epoch:{epoch}, test batch:{test_batch}, loss:{loss:.3f}, acc:{acc:.3f}")
                writer.add_scalar('test_loss', loss, global_step=test_batch)
                writer.add_scalar('test_acc', acc, global_step=test_batch)
                test_batch += 1
                test_losses.append(loss)

        # 5.3 epoch阶段的信息可视化
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        train_acc = train_true / train_total
        test_acc = test_true / test_total
        writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, global_step=epoch)
        writer.add_scalars('acc', {'train': train_acc, 'test': test_acc}, global_step=epoch)

        # 5.4 模型持久化
        if test_acc > best_acc:
            # 最优模型保存
            obj = {
                'net': net,
                'epoch': epoch,
                'train_batch': train_batch,
                'test_batch': test_batch,
                'acc': test_acc
            }
            save(obj, best_path.absolute())
            best_acc = test_acc
        if epoch % save_interval_epoch == 0:
            obj = {
                'net': net,
                'epoch': epoch,
                'train_batch': train_batch,
                'test_batch': test_batch,
                'acc': test_acc
            }
            save(obj, last_path.absolute())

    # 6. 最终模型持久化
    obj = {
        'net': net,
        'epoch': start_epoch + total_epoch - 1,
        'train_batch': train_batch,
        'test_batch': test_batch,
        'acc': test_acc
    }
    save(obj, last_path.absolute())
    writer.close()


def tt_data():
    X, Y = load_iris(return_X_y=True)
    print(type(X))
    print(type(Y))
    print(X.shape)
    print(Y.shape)

    print(X[:3])
    print(Y[:3])


def tt_model():
    best = torch.load(r"output\01\230720204537\model\best.pkl", map_location='cpu')
    last = torch.load(r"output\01\230720204537\model\last.pkl", map_location='cpu')
    print(best['epoch'], best['acc'])
    print(last['epoch'], last['acc'])


def export(model_dir):
    """
    NOTE: 可以通过netron（https://netron.app/）来查看网络结构
    将训练好的模型转换成可以支持多平台部署的结构，常用的结构：
    pt: Torch框架跨语言部署的结构
    onnx: 一种比较通用的深度学习模型框架结构
    tensorRT: 先转换成onnx，然后再进行转换使用TensorRT进行GPU加速
    openvino: 先转换为onnx，然后再进行转换使用OpenVINO进行GPU加速
    :param model_path:
    :return:
    """
    model_dir = Path(model_dir)
    # 模型恢复
    net = torch.load(model_dir / 'best.pkl', map_location='cpu')['net']
    net.eval().cpu()

    # 模型转换为pt结构
    example = torch.rand(1, 4)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save(model_dir / 'best.pt')

    # 转换为onnx结构
    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes=None  # 给定是否是动态结构
    )
    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best_dynamic.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes={
            'features': {
                0: 'batch'
            },
            'label': {
                0: 'batch'
            }
        }  # 给定是否是动态结构
    )


@torch.no_grad()
def tt_load_model(model_dir):
    model_dir = Path(model_dir)
    # pytorch的模型恢复
    net1 = torch.load(model_dir / 'best.pkl', map_location='cpu')['net']
    net1.eval().cpu()
    # pytorch script模型恢复
    net2 = torch.jit.load(model_dir / 'best.pt', map_location='cpu')
    net2.eval().cpu()
    # onnx模型恢复
    import onnxruntime
    # net3_session = onnxruntime.InferenceSession(str(model_dir / 'best.onnx'))
    net3_session = onnxruntime.InferenceSession(str(model_dir / 'best_dynamic.onnx'))

    x = torch.rand(2, 4)
    print(net1(x))
    print(net2(x))
    print(net3_session.run(['label'], input_feed={'features': x.detach().numpy()}))


if __name__ == '__main__':
    # tt_data()
    # training()
    # tt_model()
    # export(
    #     model_dir='output/01/230720204537/model'
    # )
    tt_load_model(
        model_dir='output/01/230720204537/model'
    )
