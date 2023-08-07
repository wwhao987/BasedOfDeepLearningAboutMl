import torch
import torch.nn as nn

#
# class IrisNetwork(nn.Module):
#     def __init__(self):
#         super(IrisNetwork, self).__init__()
#         self.classify = nn.Sequential(
#             nn.Linear(in_features=4, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=3)
#         )
#
#     def forward(self, x):
#         """
#         鸢尾花数据分类前向执行过程
#         :param x: [N,4] N表示N个样本，4表示每个样本有4个特征属性
#         :return: [N,3] scores N表示N个样本，3表示3个类别，每个样本属于每个类别的置信度值
#         """
#         return self.classify(x)


def tt_model():
    best = torch.load(r"output\04\230720204537\model\best.pkl", map_location='cpu')
    last = torch.load(r"output\04\230720204537\model\last.pkl", map_location='cpu')
    print(best['epoch'], best['acc'])
    print(last['epoch'], last['acc'])


if __name__ == '__main__':
    # tt_data()
    # training()
    tt_model()
