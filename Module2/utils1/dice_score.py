# import torch
# from torch import Tensor
# import os
#
# def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all batches, or for a single mask
#
#     assert input.size() == target.size()
#     if input.dim() == 2 and reduce_batch_first:
#         raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
#     # 用于计算两个张量input和target之间的相似度的。相似度的度量是两个张量的内部积除以两个张量的和，这样可以避免分母为0的情况。
#     if input.dim() == 2 or reduce_batch_first:
#         inter = torch.dot(input.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input) + torch.sum(target)
#         print("sets_sum before:", sets_sum.item())
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter
#
#         return (2 * inter + epsilon) / (sets_sum + epsilon)
#     else:
#         # compute and average metric for each batch element
#         dice = 0
#         for i in range(input.shape[0]):
#             dice += dice_coeff(input[i, ...], target[i, ...])
#         return dice / input.shape[0]
#
#
# def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all classes
#     assert input.size() == target.size()
#     dice = 0
#     for channel in range(input.shape[1]):
#         dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
#     return dice / input.shape[1]
#
#
# def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     assert input.size() == target.size()
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)



import torch
from torch import Tensor

def dice_loss(input: Tensor, target: Tensor, epsilon=1e-6):
    # Dice loss for binary classification
    assert input.size() == target.size()



    # Compute dice loss
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)

    # Handle the case where both sets are empty
    if sets_sum.item() == 0:
        sets_sum = 2 * inter

    return 1 - (2 * inter + epsilon) / (sets_sum + epsilon)
