import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        """ Implement focal loss for multi-classification 
        input : tensor, size = [batch, n_classes]
        alpha : list, data ratio, length = n_classes
        label : ground truth, size = [batch, ]
        gamma : power, int
        """
        super().__init__()
        
        self.alpha = torch.tensor(alpha, device="cuda") # size = [n_classes, 1]
        self.gamma = gamma

    def forward(self, input, target):
        """ Focal Loss = alpha * (1-p_hat)^gamma * p * log(p_hat) """
        p_hat = torch.softmax(input, 1)                # p_hat, size = [batch, n_classes]

        p_hat_power = torch.pow((1 - p_hat), self.gamma) # size = [batch, n_classes]
        log_p_hat = F.log_softmax(input, dim=1)        # size = [batch, n_classes]
        label = F.one_hot(target, num_classes=input.size(1))  # size = [batch, n_classes]
        alpha = self.alpha.expand_as(p_hat)                   # size = [batch, n_class]

        loss = -alpha * p_hat_power * label * log_p_hat

        return loss.sum(dim=1).mean()