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


# Radii for the bokeh blur model for each step
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

# Number of DIP epochs per Step
EPOCH_DICT = {
    0 : 400, 
    1 : 800, 
    2 : 1200,
    3 : 1600, 
    4 : 2000,
    5 : 2400,
    6 : 2800,
    7 : 3200, 
    8 : 3600,
    9 : 4000,
    10 : 4400,
    11 : 4800,
    12 : 5200,
    13 : 5600,
    14 : 6000,
    15 : 6400,
    16 : 6800,
    17 : 7200, 
    18 : 7600, 
    19 : 8000
}

# Downsampling sizes
DOWN_SHAPES = {
    1 : (1460, 2360),
    2 : (730, 1180),
    3 : (365, 590)
}

# Tolerance above which the DIP post-processing is started. The values gets
# higher for every few epochs, since our forward model is also less accurate
# for higher steps
DATA_TOLERANCE = {
    0 : 0.075, 
    1 : 0.075, 
    2 : 0.075,
    3 : 0.075, 
    4 : 0.075,
    5 : 0.085,
    6 : 0.085,
    7 : 0.085, 
    8 : 0.085,
    9 : 0.085,
    10 : 0.095,
    11 : 0.095,
    12 : 0.095,
    13 : 0.095,
    14 : 0.095,
    15 : 0.1,
    16 : 0.1,
    17 : 0.1, 
    18 : 0.1, 
    19 : 0.1 
}


class DIPDeblurrer(pl.LightningModule):
    def __init__(self, path_to_weights:str, step:int, lr:float=1e-4, downsampling:int=3,  which_loss:str="both",
                 kappa:float=1e-6):
        """
        Deep Image Prior deblurrer, which uses a pre-trained U-Net as initialization. If the output of the U-Net
        is above the DATA_TOLERANCE, the DIP will adapt its parameters to better fit the measurement data.
        As a forward model, a simple bokeh blur is used. The DIP is regularized by TV.

        Args:
            path_to_weights (str): Path to the general location of all U-Net weights
            step (int): Current blur step.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            downsampling (int, optional): Power 2^(x-1) of the average pooling downsampling, e.g. 
                                          downsampling=3 -> 2²=4 times spatial downsampling for
                                          the input. The output will be automatically upsampled
                                          by nearest interpolation to match the ground truth size.. Defaults to 3.
            which_loss (str, optional): Choose the loss function from "l1", "mse" and "both". Defaults to "both".
            kappa (float, optional): Regularization weight for TV. Defaults to 1e-6.
        """
        super().__init__()

        self.lr = lr
        self.step = step
        self.which_loss = which_loss
        self.downsampling = downsampling
        self.path_to_weights = path_to_weights
        self.kappa = kappa

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
            self.down = [nn.AvgPool2d(kernel_size=3, stride=2, padding=1) for i in range(self.downsampling-1)]
            self.down = nn.Sequential(*self.down)
            self.up = nn.Upsample(size=DOWN_SHAPES[1], mode='nearest')

    def set_blur(self, step):
        self.blur = BokehBlur(r=RADIUS_DICT[step]/(2**(self.downsampling-1)), shape=DOWN_SHAPES[self.downsampling])

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
        if discrepancy > DATA_TOLERANCE[self.step]:
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
            y = y + torch.randn(y.shape, device=self.device)*0.005

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
