"""
Simple U-Net for deblurring

"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from dival.reconstructors.networks.unet import get_unet_model
import numpy as np

from hdc2021_challenge.forward_model.bokeh_blur_rfft_train import BokehBlur
from hdc2021_challenge.utils.ocr import evaluateImage


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

DOWN_SHAPES = {
    1 : (1460, 2360),
    2 : (730, 1180),
    3 : (365, 590)
}


class UNetDeblurrer(pl.LightningModule):
    def __init__(self, step, lr=1e-4, scales=6, skip_channels=4, channels=None,
                 use_sigmoid=True, batch_norm=True, init_bias_zero=True,
                 downsampling=3, jittering_std=0., which_loss="l1"):
        super().__init__()

        self.lr = lr
        self.step = step
        self.jittering_std = jittering_std
        self.which_loss = which_loss

        if channels is None:
            channels = (32, 64, 128, 256, 256, 512)

        self.init_bias_zero = init_bias_zero
        self.downsampling = downsampling

        self.set_blur(step)

        save_hparams = {
            'step': step,
            'lr': lr,
            'scales': scales,
            'skip_channels': skip_channels,
            'channels': channels,
            'use_sigmoid': use_sigmoid,
            'batch_norm': batch_norm,
            'init_bias_zero': init_bias_zero,
            'downsampling': downsampling,
            'jittering_std': jittering_std,
            'which_loss': which_loss
        }
        self.save_hyperparameters(save_hparams)

        self.net = get_unet_model(
            in_ch=1, out_ch=1, scales=scales, skip=skip_channels,
            channels=channels, use_sigmoid=use_sigmoid, use_norm=batch_norm)

        if self.init_bias_zero:
            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    m.bias.data.fill_(0.0)
            self.net.apply(weights_init)

        if self.downsampling > 1:
            self.down = [nn.AvgPool2d(kernel_size=3, stride=2, padding=1) for i in range(self.downsampling-1)]
            self.down = nn.Sequential(*self.down)
            self.up = nn.Upsample(size=DOWN_SHAPES[1], mode='nearest')

    def set_blur(self, step):
        self.blur = BokehBlur(r=RADIUS_DICT[step], shape=DOWN_SHAPES[self.downsampling])

    def forward(self, y):
        if self.downsampling > 1:
            y = self.down(y)

        x = self.net(y)

        if self.downsampling > 1:
            x = self.up(x)

        return x

    def training_step(self, batch, batch_idx):
        if not isinstance(batch[0], list):
            batch = [batch]

        x, y, _ = batch[0]

        if self.downsampling > 1:
            x = self.down(x)
            y = self.down(y)

        if self.jittering_std > 0:
            y = y + torch.randn(y.shape,device=self.device)*self.jittering_std

        x_hat = self.net(y) 

        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x)
        elif self.which_loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)
           
        for i in range(1, len(batch)):
            x, _ = batch[i]
            x = x[0:2, ...]

            y = self.blur(x)

            if self.jittering_std > 0:
                y = y + torch.randn(y.shape,device=self.device)*self.jittering_std
                x = x + torch.randn(x.shape,device=self.device)*self.jittering_std

            x_hat = self.net(y) 
        
            loss = loss +  0.1 * F.mse_loss(x_hat, x)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        if batch_idx == 3:
            self.last_batch = batch

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, text = batch

        if batch_idx == 0:
            self.first_batch_val = batch

        if self.downsampling > 1:
            x = self.down(x)
            y = self.down(y)
        x_hat = self.net(y) 
        
        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x)
        elif self.which_loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)

        # preprocess image 
        x_hat = self.up(x_hat)
        x_hat = x_hat.cpu().numpy()
        x_hat = np.clip(x_hat, 0, 1)

        ocr_acc = []
        for i in range(len(text)):
            ocr_acc.append(evaluateImage(x_hat[i, 0, :, :], text[i]))

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_ocr_acc', np.mean(ocr_acc))
        return loss 

    def training_epoch_end(self, result):
        x, y, _ = self.last_batch[0]

        if self.downsampling:
            x = self.down(x)
            y = self.down(y)

        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat = self.forward(y)

            reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image(
                "deblurred", reco_grid, global_step=self.current_epoch)
            for idx in range(1, len(self.last_batch)):
                x, _ = self.last_batch[idx]
                with torch.no_grad():
                    y = self.blur(x)
                    x_hat = self.net(y) 

                    reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "deblurred set " + str(idx) , reco_grid, global_step=self.current_epoch)

                    gt_grid = torchvision.utils.make_grid(x, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "ground set " + str(idx), gt_grid, global_step=self.current_epoch)

                    blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "blurred set " + str(idx), blurred_grid, global_step=self.current_epoch)

    def validation_epoch_end(self, result):
        x, y, _ = self.first_batch_val
        if self.downsampling > 1:
            x = self.down(x)
            y = self.down(y)

        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat = self.net(y)

            reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image(
                "deblurred", reco_grid, global_step=self.current_epoch)
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
