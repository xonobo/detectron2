import torch
import torch.nn as nn
from torch.nn import functional as F

class GHMLossBase(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
    """

    def __init__(self, bins, momentum, reduction):
        super(GHMLossBase, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.reduction = reduction
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)


class SigmoidGHMLoss(GHMLossBase):

    def __init__(self, bins=10, momentum=0, reduction='none'):
        super(SigmoidGHMLoss, self).__init__(bins, momentum, reduction)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        weights = torch.zeros_like(inputs)
        p = torch.sigmoid(inputs)

        # gradient length
        g = torch.abs(p - targets)

        tot = inputs.numel()
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] \
                                      + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, weights, reduction=self.reduction) / tot

        return loss
