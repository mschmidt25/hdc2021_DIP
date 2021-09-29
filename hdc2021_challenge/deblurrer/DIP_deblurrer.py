"""
Deep Image Prior for deblurring

"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from hdc2021_challenge.forward_model.bokeh_blur_rfft_train import BokehBlur
from hdc2021_challenge.utils.ocr import evaluateImage
from hdc2021_challenge.deblurrer.UNet_deblurrer import UNetDeblurrer


RADIUS_DICT = {
    0 : 1.0*8., 
    1 : 1.2*8., 
    2 : 1.3*8.,
    3 : 1.4*8., 
    4 : 2.2*8.,
    5 : 3.75*8.,
    6 : 4.5*8.,
    7 : 5.25*8., 
    8 : 6.75*8.,
    9 : 8.2*8.,
    10 : 8.8*8.,
    11 : 9.4*8.,
    12 : 10.3*8.,
    13 : 10.8*8.,
    14 : 11.5*8.,
    15 : 12.1*8.,
    16 : 13.5*8.,
    17 : 16.0*8., 
    18 : 17.8*8., 
    19 : 19.4*8.
}

EPOCH_DICT = {
    0 : 100, 
    1 : 200, 
    2 : 300,
    3 : 400, 
    4 : 500,
    5 : 600,
    6 : 700,
    7 : 800, 
    8 : 900,
    9 : 1000,
    10 : 1100,
    11 : 1200,
    12 : 1300,
    13 : 1400,
    14 : 1500,
    15 : 1600,
    16 : 1700,
    17 : 1800, 
    18 : 1900, 
    19 : 2000
}

DATA_TOLERANCE = 0.075


class DIPDeblurrer(pl.LightningModule):
    def __init__(self, path_to_weights, step, lr=1e-4, downsampling=2,  which_loss="both",
                 kappa=1e-6, epochs=10000):
        super().__init__()

        self.lr = lr
        self.step = step
        self.which_loss = which_loss
        self.downsampling = downsampling
        self.path_to_weights = path_to_weights
        self.kappa = kappa
        self.epochs = epochs

        self.set_blur(step)

        save_hparams = {
            'step': step,
            'lr': lr,
            'downsampling': downsampling,
            'which_loss': which_loss,
            'path_to_weights': path_to_weights,
            'kappa': kappa
        }
        self.save_hyperparameters(save_hparams)

        self.set_network(step)

        if self.downsampling > 1:
            self.down = [nn.AvgPool2d(kernel_size=3, stride=2, padding=1) for i in range(self.downsampling)]
            self.down = nn.Sequential(*self.down)
            self.up = nn.Upsample(size=(1460, 2360), mode='nearest')

    def set_blur(self, step):
        self.blur = BokehBlur(r=RADIUS_DICT[step]/(2**self.downsampling), shape=(365, 590))

    def set_network(self, step):
        path = self.path_to_weights + 'step_' + str(step) + ".ckpt"
        print('Loading network weights from: ' + path)
        initial_reconstructor = UNetDeblurrer.load_from_checkpoint(path)
        self.net = initial_reconstructor.net

    def forward(self, y):
        # Downsample measurement to the desired level
        if self.downsampling > 1:
            y = self.down(y)
            
        if self.which_loss == 'l1' or self.which_loss == 'both':
                l1_loss = torch.nn.L1Loss()

        # Initial checkup
        x_hat = self.net(y)
        y_hat = self.blur(x_hat)

        if self.which_loss == 'l1':
            discrepancy = l1_loss(y_hat, y)
        elif self.which_loss == 'both':
            discrepancy = 0.5*l1_loss(y_hat, y) + 0.5*F.mse_loss(y_hat, y)
        else: 
            discrepancy = F.mse_loss(y_hat, y)
        discrepancy = discrepancy.detach().cpu().numpy()

        # DIP Postprocessing (if necessary)
        if discrepancy > DATA_TOLERANCE:
            print('The data discrepancy is above the tolerance: ' + str(discrepancy))
            print('Starting DIP post-processing...')

            dip_optimizer = self.configure_optimizers()

            self.net.train()
            for i in tqdm(range(EPOCH_DICT[self.step])):
                with torch.set_grad_enabled(True):
                    dip_optimizer.zero_grad()

                    # Calculate DIP step
                    x_hat = self.net(y)
                    y_hat = self.blur(x_hat)

                    # Update DIP
                    if self.which_loss == 'l1':
                        loss = l1_loss(y_hat, y)
                    elif self.which_loss == 'both':
                        loss = 0.5*l1_loss(y_hat, y) + 0.5*F.mse_loss(y_hat, y)
                    else: 
                        loss = F.mse_loss(y_hat, y)
                    
                    loss = loss + self.kappa*tv_loss(x_hat)
                    loss.backward()
                    dip_optimizer.step()
            print('DIP Postprocessing complete.')
            self.net.eval()
        else:
            print('Initial output within tolerance. No postprocessing needed.')

        # Upsample measurement to the original size
        if self.downsampling > 1:
            x_hat = self.up(x_hat)

        return x_hat

    def training_step(self, batch, batch_idx):
        # Training on challenge data or STL10
        if len(batch) == 3:
            x, y, text = batch
            if self.downsampling > 1:
                x = self.down(x)
                y = self.down(y)
        else:
            x, _ = batch
            x = self.down(x)
            y = self.blur(x)

        x_hat = self.net(y)
        y_hat = self.blur(x_hat)

        # Calculate data discrepancy
        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(y_hat, y)
        elif self.which_loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(y_hat, y) + 0.5*F.mse_loss(y_hat, y)
        else: 
            loss = F.mse_loss(y_hat, y)

        # Add regularization
        loss = loss + self.kappa*tv_loss(x_hat)
        self.log('train_loss', loss)

        # Calculate OCR accuracy
        if len(batch) == 3:
            x_ocr = self.up(x_hat)
            x_ocr = x_ocr.detach().cpu().numpy()
            x_ocr = np.clip(x_ocr, 0, 1)

            ocr_acc = []
            for i in range(len(text)):
                ocr_acc.append(evaluateImage(x_ocr[i, 0, :, :], text[i])+
                            self.global_step*1e-6)
            self.log('train_ocr_acc', np.mean(ocr_acc))
        else:
            train_mse = F.mse_loss(x_hat, x)
            self.log('train_mse', train_mse)
        
        # Logging to TensorBoard
        if self.global_step == 0:
            img_grid = torchvision.utils.make_grid(x, normalize=True,
                                                scale_each=True)
            self.logger.experiment.add_image(
                "ground truth", img_grid, global_step=self.current_epoch)

            blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                                scale_each=True)
            self.logger.experiment.add_image(
                "blurred image", blurred_grid, global_step=self.current_epoch)

        reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                            scale_each=True)
        self.logger.experiment.add_image(
            "deblurred", reco_grid, global_step=self.current_epoch)

        forward_grid = torchvision.utils.make_grid(y_hat, normalize=True,
                                                            scale_each=True)
        self.logger.experiment.add_image(
            "forward", forward_grid, global_step=self.current_epoch)

        return loss
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def tv_loss(x):
    """
    Isotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])
