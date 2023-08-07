import numpy as np

from code0724.iris_processor import IrisProcessor

processor = IrisProcessor("../code0720/output/01/230720204537/model/best_dynamic.onnx")

while True:
    x = input("请输入特征属性，使用空格隔开:")
    if "q" == x:
        break
    x = x.split(" ")
    if len(x) != 4:
        print(f"输入特征属性异常，请输入4维特征属性:{x}")
        continue
    x = np.asarray([x])  # [1,4]
    r = processor.predict(x)
    print(f"预测结果为:{r}")
