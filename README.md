# LimboNet
- Official Repository for the 1st place of NYUSH CS360 Machine Learning Final Competition in Fall 2023


LimboNet is a custom neural network model which is a variant of the ResNeXt and InceptionNet architecture. It is named after the "Limbo" in the movie INCEPTION consists of several stages, each implemented using the CobbBottleneck module.

The CobbBottleneck class is a modified version of the bottleneck structure of ResNeXt models, mainly perserve its residual connection with cardinality, and Multimodal Convolutional layers in InceptionNet V2. It consists of several convolutional layers with different kernal size with maxpool, batch normalization and ReLU activations. The whole structure is shown in the next page.

# Model Structure

<img width="283" alt="截屏2023-12-17 09 05 27" src="https://github.com/Exiam6/LimboNet/assets/121872598/363904d4-ed80-455c-b5f9-9fc3ddfc52f1">

> The Structure of LimboNet

<img width="797" alt="Block" src="https://github.com/Exiam6/LimboNet/assets/121872598/a8a0ad25-9cfd-4dec-acfd-74445072823e">

> The Structure of Basic Blocks in LimboNet. (2 computational layers each)

![LimboNet](https://github.com/Exiam6/LimboNet/assets/121872598/b506ea3f-1323-4919-975d-f78315dd6a73)
> The forward and backward path in LimboNet
