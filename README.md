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
This method combines the regularized Deep Image Prior (DIP) approach with a pre-trained U-Net model $$F_\theta$$. The U-Net acts as a fully-learned inversion model $$F:Y \rightarrow x$$ and is trained on the blurry text images. The downside of such fully-learned models (for inversion or post-processing) is the missing data consistency. The models will generally perform well on data that is similiar to the distribution of the training samples, but out-of-distribution samples can severely harm the model's performance. Therefore, we introduce a second step, where we use the formulation of the Deep Image Prior to adapt the weights of the U-Net in case of missing data consistency. Using pre-trained weights for the DIP also has the advantage, that it will need less iterations to converge. To check the data consistency, we need a forward model. In our case, we approximate the blurring operation with a convolution.

Overall the reconstruction works as follows:
1. Create first reconstruction with the U-Net model: $\hat{x} = F_\theta(y^\delta)$
2. Check data consistency $\mathcal{D}$ of the initial reconstruction: $\mathcal{D}(\mathcal{A}\hat{x}, y^\delta)$
3. If the data consistency is below a certain threshold, we accept the reconstruction $\hat{x} $.
4. If the data consistency is above the threshold, we adapt the weights of the U-Net with the DIP method: $\hat{\theta} = arg\min_\theta \mathcal{D}(\mathcal{A}F_\theta(y^\delta), y^\delta) + \kappa TV(F_\theta(y^\delta))$
5. In case of the DIP post-processing of the weights, the final reconstruction is given by: $\hat{x} = F_\hat{\theta}(y^\delta)$

### Approximate Forward Model 
For a fixed blurring level the out-of-focus blur can be modeled by as a linear, position invariant convolution with a circular point-spread-function: 

<a href="https://www.codecogs.com/eqnedit.php?latex=g_\eta&space;=&space;\mathcal{A}&space;f&space;&plus;&space;\eta&space;=&space;k&space;*&space;f&space;&plus;&space;\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_\eta&space;=&space;\mathcal{A}&space;f&space;&plus;&space;\eta&space;=&space;k&space;*&space;f&space;&plus;&space;\eta" title="g_\eta = \mathcal{A} f + \eta = k * f + \eta" /></a>
with 

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x)&space;=&space;\left\{\begin{array}{lr}&space;\frac{1}{\pi&space;r^2},&space;&&space;\text{for&space;}&space;\|&space;x&space;\|^2&space;\le&space;r^2\\&space;0,&space;&&space;\text{else&space;}&space;\end{array}\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x)&space;=&space;\left\{\begin{array}{lr}&space;\frac{1}{\pi&space;r^2},&space;&&space;\text{for&space;}&space;\|&space;x&space;\|^2&space;\le&space;r^2\\&space;0,&space;&&space;\text{else&space;}&space;\end{array}\right\}" title="k(x) = \left\{\begin{array}{lr} \frac{1}{\pi r^2}, & \text{for } \| x \|^2 \le r^2\\ 0, & \text{else } \end{array}\right\}" /></a>

This model works well for small blurring levels. For higher blurring levels the average error between the approximate model and the real measurements gets bigger.

### DIP
The Deep Image Prior is an unsupervised reconstruction method, which works with a single measurement. It uses a forward model to simulate the measurement process and allow for a comparison with the noisy measurement. Based on this information, the goal is to adapt (often randomly initialized) network weights $\theta$ to minimize the data discrepancy $\mathcal{D}$ for a fixed network input $z$. In our case, we use the noisy measurement $y^\delta$ as the fixed input and the pre-trained U-Net weights for the initial parameter choice of the DIP network. In addition, the problem is regularized by a weighted total variation (TV) term to avoid overfitting to the noise. The DIP minimization problem is:

$$\hat{theta} arg\min_\theta \mathcal{D}(\mathcal{A}F_\theta(y^\delta), y^\delta) + \kappa TV(F_\theta(y^\delta))$$

In general, this is minimized by gradient descent methods, e.g. ADAM in our experiments. Finally, the best parameters $\hat{theta}$ are used to create the reconstruction $\hat{x} = F_\hat{\theta}(y^\delta)$.

### Data Consistency Threshold
The data consistency threshold manages, if the DIP adaption of the U-Net is necessary for the current sample. The threshold was empirically chosen based on data consistency values from in-distribution samples and the error of our approximate forward model. Since our forward model gets worse for higher blur steps, we slightly increase the threshold every 5 blur steps.


### Training
A seperate U-Net model for each blurring step was trained on the blurry text images. The DIP does not need training.


### Reference results
OCR accuracy on our test set (20 images per step):
- 0: 81.60
- 1: 86.40
- 2: 81.50
- 3: 92.45
- 4: 88.90
- 5: 80.80
- 6: 84.70
- 7: 83.95
- 8: 79.75
- 9: 74.40
- 10: 77.80
- 11: 70.25
- 12: 73.20
- 13: 67.65
- 14: 78.60
- 15: 57.65
- 16: 45.70
- 17: 40.70
- 18: 29.25
- 19: 26.15

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
