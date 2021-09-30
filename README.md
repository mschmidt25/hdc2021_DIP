# HDC 2021 DIP
One of the submissions from the ZeTeM Uni Bremen Team for the Helsinki Deblur Challenge 2021 (HDC 2021).
https://www.fips.fi/HDC2021.php

Team members are listed below.

## Requirements
The main requirements for our code are listed below. You can also use the requirements.txt file to replicate our conda environment.
* numpy = 1.20.3
* pytorch = 1.9.0
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0
* dival = 0.6.1
* torchvision = 0.10.0
* pytesseract = 0.3.8
* fuzzywuzzy = 0.18.0

## Install
Install the package using:

```
pip install -e .
```
Make sure to have git-lfs installed to pull the weight files for the model.

Please download the weights from this link:
https://seafile.zfn.uni-bremen.de/d/a90a26b1721b461db30a/
and place them in "weights/hdc2021_challenge/weights/deblurring/unet_deblurring"

## Usage
Prediction on images in a folder can be done using:

```
python hdc2021_challenge/main.py path-to-input-files path-to-output-files step
```

If you want to use the other methods for training etc., change the BATH_PATH in data_util.py to the directory of the HDC Challenge data. 

## Method


### Reconstruction


### Training


### Reference results
OCR accuracy on our test set (20 images per step):
- 0:
- 1:
- 2:
- 3:
- 4:
- 5:
- 6:
- 7:
- 8:
- 9:
- 10:
- 11:
- 12:
- 13:
- 14:
- 15:
- 16:
- 17:
- 18:
- 19:

## Authors
Team University of Bremen, Center of Industrial Mathematics (ZeTeM) et al.:
- Alexander Denker (Bremen)
- Maximilian Schmidt (Bremen)
- Johannes Leuschner (Bremen)
- Sören Dittmer (Bremen/Cambridge)
- Judith Nickel (Bremen)
- Clemens Arndt (Bremen)
- Gael Rigaud (Bremen)
- Richard Schmähl (Stuttgart)

## Examples
Random reconstructions from the test set on different blur steps. For each
reconstruction, the OCR accuracy on the middle text line is reported.

### Step 2
![Blur step 2](example_images/step_2test_sample6.png "Step 2")

### Step 5
![Blur step 5](example_images/step_5test_sample15.png "Step 5")

### Step 10
![Blur step 10](example_images/step_10test_sample8.png "Step 10")

### Step 12
![Blur step 12](example_images/step_12test_sample7.png "Step 12")

### Step 15
![Blur step 15](example_images/step_15test_sample8.png "Step 15")

### Step 19
![Blur step 19](example_images/step_19test_sample12.png "Step 19")
