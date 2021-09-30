"""
Evaluate the DIP OCR performance on the test set.
"""

import os
from pathlib import Path

import yaml
from dival.measure import PSNR, SSIM
from tqdm import tqdm
import numpy as np
from dival.util.plot import plot_images
import matplotlib.pyplot as plt 

from hdc2021_challenge.utils.ocr import evaluateImage
from hdc2021_challenge.utils.blurred_dataset import BlurredDataModule
from hdc2021_challenge.deblurrer.DIP_deblurrer import DIPDeblurrer


# General path to the U-Net weights
base_path = os.path.join(os.path.dirname(__file__), '..')
experiment_name = 'deblurring' 
version = 'unet_deblurring'
chkp_name = ''
path_parts = [base_path, 'weights', experiment_name, version, chkp_name]
path_to_weights = os.path.join(*path_parts)

for step in range(20):
    print("Eval OCR for step ", step)
    print("--------------------------------\n")
    save_report = True 

    reconstructor = DIPDeblurrer(path_to_weights=path_to_weights,
                             step=step,
                             lr=1e-4,
                             downsampling=3,
                             which_loss="both",
                             kappa=1e-6)
    reconstructor.to("cuda")
    
    if save_report:
        report_name = 'dip' + '_step=' + str(step) + "_ocr"
        report_path = ['results']
        report_path.append(report_name)
        report_path = os.path.join(*report_path)
        Path(report_path).mkdir(parents=True, exist_ok=True)

    dataset = BlurredDataModule(batch_size=1, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()

    num_test_images = len(dataset.test_dataloader().dataset)

    psnrs = []
    ssims = []
    ocr_acc = []

    for i, batch in tqdm(zip(range(num_test_images), dataset.test_dataloader()),
                             total=num_test_images):
        # Reset network to initial weights
        reconstructor.set_network(step)
        reconstructor.to("cuda")
        
        # Prepare input data
        gt, obs, text = batch
        obs_cpu = obs
        obs = obs.to('cuda')

        # Create reconstruction from observation
        reco = reconstructor.forward(obs)
        reco = reco.detach().cpu().numpy()
        reco = np.clip(reco, 0, 1)
        
        # Calculate quality metrics
        psnrs.append(PSNR(reco[0][0], gt.numpy()[0][0]))
        ssims.append(SSIM(reco[0][0], gt.numpy()[0][0]))
        ocr_acc.append(evaluateImage(reco[0][0], text[0]))

        print(text)
        print(ocr_acc[-1])

        if ocr_acc[-1] < 70.:
            _, ax = plot_images([obs_cpu[0,0,:,:], reco[0,0,:,:].T, gt[0,0,:,:].T],
                                    fig_size=(10, 4), vrange='equal', cbar='auto')
            ax[0].set_title('Measurement')
            ax[1].set_title('Reconstruction - ' + str(ocr_acc[-1]))
            ax[2].set_title('Ground truth')
            
            if save_report:
                img_save_path = os.path.join(report_path,'img')

                Path(img_save_path).mkdir(parents=True, exist_ok=True)
                img_save_path = os.path.join(img_save_path, 'step_' + str(step) + 'test sample' + str(i) + '.png')
                plt.savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=None, transparent=False,
                        bbox_inches=None, pad_inches=0.1, metadata=None)

    mean_psnr = np.mean(psnrs)
    std_psnr = np.std(psnrs)
    mean_ssim = np.mean(ssims)
    std_ssim = np.std(ssims)

    print('---')
    print('Results:')
    print('mean psnr: {:f}'.format(mean_psnr))
    print('std psnr: {:f}'.format(std_psnr))
    print('mean ssim: {:f}'.format(mean_ssim))
    print('std ssim: {:f}'.format(std_ssim))
    print('mean ocr acc: ', np.mean(ocr_acc))

    if save_report:
        report_dict = {'settings': {'num_test_images': num_test_images},
                    'results': {'mean_psnr': float(np.mean(psnrs)) , 
                                'std_psnr': float(np.std(psnrs)),
                                'mean_ssim': float(np.mean(ssims)) ,
                                'std_ssim': float(np.std(ssims)), 
                                'mean_ocr_acc': float(np.mean(ocr_acc)) }}
        report_file_path =  os.path.join(report_path, 'report.yaml')
        with open(report_file_path, 'w') as file:
            documents = yaml.dump(report_dict, file)
