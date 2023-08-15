import torch
import torch.nn as nn

class BCELossWithLogits(nn.Module):
    def __init__(self):
        super(BCELossWithLogits, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        return self.bce_loss(outputs, labels.float())