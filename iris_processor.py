"""
Iris预测模型处理器代码
"""
import os

import numpy as np
import onnxruntime
import torch


def softmax(scores):
    """
    求解softmax概率值
    :param scores: numpy对象 [n,m]
    :return: 求解属于m个类别的概率值
    """
    a = np.exp(scores)
    b = np.sum(a, axis=1, keepdims=True)
    p = a / b
    return p


class IrisProcessor(object):
    def __init__(self, model_path):
        """
        模型初始化
        :param model_path: 对应的模型文件路径，支持pt和onnx后缀
        """
        super(IrisProcessor, self).__init__()
        model_path = os.path.abspath(model_path)
        _, ext = os.path.splitext(model_path.lower())
        self.pt, self.onnx = False, False
        if ext == '.pt':
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval().cpu()
            self.model = model
            self.pt = True
        elif ext == '.onnx':
            session = onnxruntime.InferenceSession(model_path)
            self.session = session
            self.input_name = 'features'
            self.output_name = 'label'
            self.onnx = True
        else:
            raise ValueError(f"当前仅支持pt和onnx格式，当前文件类型为:{model_path}")
        self.classes = ['类别1', '类别2', '类别3']
        print(f"模型恢复成功:pt->{self.pt} ; onnx->{self.onnx}")

    def _process_after_model(self, x, scores):
        """
        后处理逻辑
        :param x: 原始的特征属性x numpy类型 [n,4]
        :param scores: 模型预测的置信度信息 numpy类型 [n,3]
        :return: 每个样本均返回对应的预测类别名称、id以及概率值，以dict形式返回
        """
        pred_probas = softmax(scores)  # [n,3] -> [n,3]
        pred_indexes = np.argmax(scores, axis=1)  # [n,3] -> [n]
        result = []
        for k, idx in enumerate(pred_indexes):
            r = {
                'id': int(idx),  # 将numpy的int类型转换为python的int类型
                'label': self.classes[idx],
                'proba': float(pred_probas[k][idx])  # 将numpy的float类型转换成python的float类型
            }
            result.append(r)
        return result

    @torch.no_grad()
    def _predict_with_pt(self, x):
        tensor_x = torch.from_numpy(x).to(torch.float)
        scores = self.model(tensor_x)  # [n,4] -> [n,3]
        scores = scores.numpy()  # tensor -> numpy
        return self._process_after_model(x, scores)

    def _predict_with_onnx(self, x):
        onnx_x = x.astype('float32')
        # session.run会返回output_names给定的每个名称对应的预测结果，最终是一个list列表，列表大小和参数output_names大小一致
        scores = self.session.run(
            output_names=[self.output_name],
            input_feed={self.input_name: onnx_x}
        )  # [n,4] -> list([n,3])
        scores = scores[0]  # 获取第一个输出结果（output_name对应结果）
        return self._process_after_model(x, scores)

    def predict(self, x):
        """
        模型预测方法，输入鸢尾花的原始特征属性，返回对应的预测标签
        :param x: numpy对象，形状为[n,4]表示n个样本，4个属性
        :return: 每个样本均返回对应的预测类别名称、id以及概率值，以dict形式返回
        """
        if self.pt:
            return self._predict_with_pt(x)
        elif self.onnx:
            return self._predict_with_onnx(x)
        else:
            raise ValueError("当前模型初始化异常!")
