import numpy as np
import torch
import torch.nn as nn


def sqrt(n):
    return np.sqrt(n)


if __name__ == '__main__':
    z = (
        1 / sqrt(1 ** 2 + 9 ** 2),
        9 / sqrt(1 ** 2 + 9 ** 2 + 5 ** 2),
        5 / sqrt(9 ** 2 + 5 ** 2 + 7 ** 2),
        7 / sqrt(5 ** 2 + 7 ** 2 + 9 ** 2),
        9 / sqrt(7 ** 2 + 9 ** 2 + 2 ** 2),
        2 / sqrt(9 ** 2 + 2 ** 2 + 4 ** 2),
        4 / sqrt(2 ** 2 + 4 ** 2)
    )
    print(z)

    lrn = nn.LocalResponseNorm(size=7, k=1, alpha=7)
    x = torch.tensor([[[1], [9], [1], [1], [1], [1], [1], [5], [1], [1], [9], [5], [1], [1], [1], [1], [1], [5], [1], [5], [7], [9], [2], [4]]])  # [1,7,1] C=7
    print(lrn(x))
