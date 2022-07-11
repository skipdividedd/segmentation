import torch
import numpy as np
import torch.optim
from time import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from metrics import iou_pytorch
from inference import score_model
from dataset import Dataset
from loss import BCEWithLogits, dice_loss, focal_loss

from models.model_segnet import SegNet
from models.model_unet import UNet

data_tr, data_val, data_ts = Dataset()


def train_model(model, opt, loss_fn, epochs, data_tr, data_val):
    device = 'cpu'
    X_val, Y_val = next(iter(data_val))

    history = []

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        avg_loss_tr = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # set parameter gradients to zero
            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate loss to show the user
            avg_loss_tr += loss / len(data_tr)


        toc = time()

        # show intermediate results
        model.eval()  # testing mode
        avg_loss_val = 0
        for X_val, Y_val in data_val:
            with torch.no_grad():
                Y_hat = model(X_val.to(device))

        loss = loss_fn(Y_hat, Y_val)
        avg_loss_val += loss / len(data_val)
        print('loss: %f' % avg_loss_val)

        avg_score_val = score_model(model, iou_pytorch, data_val)
        print('score: %f' % avg_score_val)

        history.append((avg_loss_val, avg_score_val))

        # Visualize tools
        clear_output(wait=True)
        for k in range(5):
            plt.subplot(2, 5, k + 1)
            plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 5, k + 6)
            plt.imshow(Y_hat[k, 0] > 0.5, cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss_tr: %f' % (epoch + 1, epochs, avg_loss_tr))  # TRAIN LOSS
        plt.show()

    return history


def run_experiment(model, loss):
    device = 'cpu'
    if model == 'Segnet':
        if loss == 'BCEWithLogits':
            model = SegNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss = BCEWithLogits
            max_epochs = 10
            history = train_model(model, optimizer, loss,
                                  max_epochs, data_tr, data_val)

        elif loss == 'Dice':
            model_dice = SegNet().to(device)
            max_epochs = 13
            optimizer = torch.optim.Adam(model_dice.parameters(), lr=1e-4)
            history = train_model(model_dice, optimizer, dice_loss,
                                  max_epochs, data_tr, data_val)

        elif loss == 'Focal':
            model_focal = SegNet().to(device)
            max_epochs = 10
            optimizer = torch.optim.Adam(model_focal.parameters(), lr=1e-4)
            history = train_model(model_focal, optimizer, focal_loss,
                                  max_epochs, data_tr, data_val)

    elif model == 'Unet':
        if loss == 'BCEWithLogits':
            unet_model_bce = UNet().to(device)
            optim = torch.optim.Adam(unet_model_bce.parameters(), lr=1e-4)
            max_epochs = 12
            loss = BCEWithLogits
            history = train_model(unet_model_bce, optim, loss,
                                  max_epochs, data_tr, data_val)

        elif loss == 'Focal':
            unet_model_focal = UNet().to(device)
            optimizer = torch.optim.Adam(unet_model_focal.parameters(), lr=1e-4)
            max_epochs = 6
            loss = focal_loss
            history = train_model(unet_model_focal, optimizer, loss,
                                  max_epochs, data_tr, data_val)

    return history


#run_experiment('Unet', 'Focal')