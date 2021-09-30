"""
The Deep Image Prior does not require a training. Nonetheless, it can be helpful
to evaluate the number of epochs needed to create convincing results. In this
script, one can train the model on one image of the deblur or STL10 data.
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms

from hdc2021_challenge.utils.blurred_dataset import BlurredDataModule
from hdc2021_challenge.deblurrer.DIP_deblurrer import DIPDeblurrer


# Basic training parameters
start_step = 0
data = 'stl10'
stl_root = "/localdata/STL10"
download_stl = False
downsampling = 3
epochs = 5

# General path to the U-Net weights
base_path = os.path.join(os.path.dirname(__file__), '..')
experiment_name = 'deblurring' 
version = 'unet_deblurring'
chkp_name = ''
path_parts = [base_path, 'weights', experiment_name, version, chkp_name]
path_to_weights = os.path.join(*path_parts)

# "Train" on all 20 steps
for step in range(start_step, 20):
    # Configure the reconstructor
    reconstructor = DIPDeblurrer(path_to_weights=path_to_weights,
                                step=step,
                                lr=1e-4,
                                downsampling=downsampling,
                                which_loss="both",
                                kappa=1e-6)

    # Load the dataset
    if data == 'deblur':
        dataset = BlurredDataModule(batch_size=1, blurring_step=step,
                                    num_data_loader_workers=0)
        dataset.prepare_data()
        dataset.setup()
        dataloader = DataLoader(dataset.blurred_dataset_validation, batch_size=1,
                                num_workers=0, shuffle=False, pin_memory=True)

        checkpoint_callback = ModelCheckpoint(dirpath=None,
                                        save_top_k=1,
                                        verbose=True,
                                        monitor='train_ocr_acc',
                                        mode='max')

    elif data == 'stl10':
        transform_stl10 = transforms.Compose(
        [transforms.Grayscale(), 
        transforms.ToTensor(), 
        transforms.Resize(size=(1460, 2360))])

        trainset_stl10 =  torchvision.datasets.STL10(root=stl_root, split='train',
                                                     download=download_stl, transform=transform_stl10)
        dataloader = DataLoader(trainset_stl10, batch_size=1, shuffle=False, num_workers=0)

        checkpoint_callback = ModelCheckpoint(dirpath=None,
                                        save_top_k=1,
                                        verbose=True,
                                        monitor='train_mse',
                                        mode='min')
    else:
        NotImplementedError()

    # Folder for storing weights and the tensorboard log
    base_path = 'deblurring_experiments'
    experiment_name = 'dip_deblurring'
    blurring_step = "step_" + str(step)
    path_parts = [base_path, experiment_name, blurring_step]
    log_dir = os.path.join(*path_parts)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

    # Arguments for the Pytorch Lightning trainer
    trainer_args = {'gpus': [0],
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback],
                    'benchmark': False,
                    'fast_dev_run': False,
                    'gradient_clip_val': 1.0,
                    'logger': tb_logger,
                    'log_every_n_steps': 1,
                    'limit_train_batches': 1}

    # Train the model
    trainer = pl.Trainer(max_epochs=epochs, **trainer_args)
    trainer.fit(reconstructor, train_dataloader=dataloader)
