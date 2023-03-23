import torch
import numpy as np
from torch import Tensor 
import torch.nn.functional as F


class ApproximateKLLoss(torch.nn.Module):
    # KL Divergence loss without the evidence part (log sum exp) 
    def __init__(self, weight=None, size_average=True):
        super(ApproximateKLLoss, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        temperature = 1.0 / 5.0

        entropy = torch.sum(F.softmax(inputs * temperature, dim=1) * F.log_softmax(inputs * temperature, dim=1), dim=0)  # H

        # cross entropy part
        p1j = F.softmax(inputs * temperature, dim=1)
        Tz2j = temperature * targets
        crossentropy = torch.sum(p1j * Tz2j, dim=0)  # CE

        loss = entropy - crossentropy
        loss = loss.sum() / inputs.size(0)
        return loss
