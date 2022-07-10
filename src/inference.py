import torch
import torch.optim


def score_model(model, metric, data):
    model.eval()  # testing mode
    scores = 0
    device = 'cpu'
    for X_batch, Y_label in data:
        Y_pred = (model(X_batch.to(device)).sigmoid() > 0.5).type(torch.float)
        scores += metric(Y_pred, Y_label.to(device)).mean().item()

    return scores/len(data)