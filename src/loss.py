import torch
import torch.optim


def BCEWithLogits(y_pred, y_true):
    bcelogits_loss = (y_pred - y_true * y_pred + torch.log(1 + torch.exp(-y_pred))).mean()
    return bcelogits_loss


def dice_loss(y_pred,y_real):
    eps = 1e-8
    y_pred = torch.clamp(torch.sigmoid(y_pred), min=eps, max=1-eps)
    X = y_pred.view(-1)
    Y = y_real.view(-1)
    intersection = (X * Y).sum()
    return 1 - (1/65536) * (2.*intersection + eps)/(X.sum() + Y.sum() + eps)


def focal_loss(inputs, targets, eps = 1e-8, gamma = 2):
    alpha = 0.8
    inputs = torch.clamp(torch.sigmoid(inputs), min=eps, max=1-eps)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    BCE = (inputs - targets * inputs + torch.log(1 + torch.exp(-inputs))).mean()
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
    return focal_loss

