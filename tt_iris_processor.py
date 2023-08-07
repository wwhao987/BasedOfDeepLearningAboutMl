import numpy as np

from code0724.iris_processor import IrisProcessor

if __name__ == '__main__':
    # processor = IrisProcessor("../code0720/output/01/230720204537/model/best.pt")
    processor = IrisProcessor("../code0720/output/01/230720204537/model/best_dynamic.onnx")
    r = processor.predict(np.asarray([[5, 2.3, 1.5, 2.2], [0.2, 1.3, 0.5, 0.2]]))
    print(r)
