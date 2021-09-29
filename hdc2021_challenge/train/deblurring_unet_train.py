import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from hdc2021_challenge.utils.blurred_dataset import MultipleBlurredDataModule, BlurredDataModule
from hdc2021_challenge.deblurrer.UNet_deblurrer import UNetDeblurrer


start_step = 0
epochs = 100
batch_size = 16
downsampling = 3
multi_data = False

reconstructor = UNetDeblurrer(step=start_step,
                            lr=1e-4,
                            scales=6,
                            skip_channels=16,
                            channels=(32, 64, 128, 256, 256, 512),
                            use_sigmoid=True,
                            batch_norm=True,
                            init_bias_zero=True,
                            downsampling=downsampling,
                            jittering_std=0.005,
                            which_loss='mse')

for step in range(start_step, 20):

    if multi_data:
        reconstructor.set_blur(step)
        dataset = MultipleBlurredDataModule(batch_size=batch_size, blurring_step=step,
                                            img_size=(int(1460 // (2**downsampling)),
                                                    int(2360 // (2**downsampling) - 1)),
                                            num_data_loader_workers=0)
    else:
        dataset = BlurredDataModule(batch_size=batch_size, blurring_step=step,
                                    num_data_loader_workers=8)
    dataset.prepare_data()
    dataset.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        verbose=True,
        monitor='val_ocr_acc',
        mode='max',
    )

    base_path = 'deblurring_experiments'
    experiment_name = 'unet_deblurring'
    blurring_step = "step_" + str(step)
    path_parts = [base_path, experiment_name, blurring_step]
    log_dir = os.path.join(*path_parts)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

    trainer_args = {'plugins': DDPPlugin(find_unused_parameters=True),
                    'gpus': -1,
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback],
                    'benchmark': False,
                    'fast_dev_run': False,
                    'gradient_clip_val': 1.0,
                    'logger': tb_logger,
                    'log_every_n_steps': 20,
                    'auto_scale_batch_size': 'binsearch',
                    'multiple_trainloader_mode': 'min_size'}

    trainer = pl.Trainer(max_epochs=epochs, **trainer_args)

    trainer.fit(reconstructor, datamodule=dataset)
    reconstructor = reconstructor.load_from_checkpoint(checkpoint_callback.best_model_path)
