# hdc2021_DIP (In Progress)
One of our submissions for the HDC2021. 

## Install 

Install the package using:

```
pip install -e .
```

Please download the weights from the following link: 
https://seafile.zfn.uni-bremen.de/d/a90a26b1721b461db30a/
Afterwards, move them to "hdc2021_challenge/weights/deblurring/unet_deblurring/"

## Usage 

Prediction on images in a folder can be done using:

```
python hdc2021_challenge/main.py path-to-input-files path-to-output-files step
```

## Method

TODO

## Examples

## Requirements 

* numpy = 1.20.3
* pytorch = 1.9.0 
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0
* dival = 0.6.1

## Authors

Team University of Bremen, Center of Industrial Mathematics (ZeTeM) et al.: 
- Alexander Denker, Maximilian Schmidt, Johannes Leuschner, Sören Dittmer, Judith Nickel, Clemens Arndt, Gael Rigaud, Richard Schmähl
