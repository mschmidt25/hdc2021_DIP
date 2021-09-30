import os
import argparse
from PIL import Image

import torch 
import matplotlib.pyplot as plt 
import numpy as np 

from hdc2021_challenge.deblurrer.DIP_deblurrer import DIPDeblurrer


parser = argparse.ArgumentParser(description='Apply DIP Deblurrer to every image in a directory.')
parser.add_argument('input_files')
parser.add_argument('output_files')
parser.add_argument('step')


def main(input_files, output_files, step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to U-Net weights
    base_path = os.path.join(os.path.dirname(__file__), 'weights')
    experiment_name = 'deblurring' 
    version = 'unet_deblurring'
    chkp_name = ''
    path_parts = [base_path, 'weights', experiment_name, version, chkp_name]
    path_to_weights = os.path.join(*path_parts)

    # Create reconstructor
    reconstructor = DIPDeblurrer(path_to_weights=path_to_weights,
                             step=int(step),
                             lr=1e-4,
                             downsampling=3,
                             which_loss="both",
                             kappa=1e-6)
    reconstructor.to(device)

    for f in os.listdir(input_files):
        if f.endswith("tif"):
            y = np.array(Image.open(os.path.join(input_files, f)))
            print(y.shape)
            y = torch.from_numpy(y/65535.).float()
            y = y.unsqueeze(0).unsqueeze(0)
            y = y.to(device)
            
            # Reset weights of the DIP
            reconstructor.set_network(int(step))
            reconstructor.to(device)

            x_hat = reconstructor.forward(y)
            x_hat = x_hat.detach().cpu().numpy()

            im = Image.fromarray(x_hat[0][0]*255.).convert("L")
            print(im)
            os.makedirs(output_files, exist_ok=True)
            im.save(os.path.join(output_files,f.split(".")[0] + ".PNG"))

    return 0


if __name__ == "__main__":

    args = parser.parse_args()
    main(args.input_files, args.output_files, args.step)
