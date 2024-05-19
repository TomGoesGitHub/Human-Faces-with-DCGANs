# Creating Artificial Human Faces with Generative Adversarial Networks

## Abstract
This project explores the image generation of realistic human faces using Generative Adversarial Networks. I describe the theory of the overall framework and summarize the
insights of my personal learning process. Finally, leveraging a range of best-practice stabilization techniques, a from-scratch implemented model is used to synthesize diverse
and lifelike facial images from latent space representations. 

## Demo
(GIF may take a few seconds to load)

Generated Faces during the training process.
![training_dcgan_on_celeba](https://github.com/TomGoesGitHub/Human-Faces-with-DCGANs/assets/81027049/5799a0ef-f002-4b02-ab9b-b0b522530adf)

## Dataset
Celebrity Faces Attributes Dataset (CelebA) was used for training. It is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.
The images in this dataset cover large pose variations and background clutter. To minimize model complexity, in this project the images were downsized to 64×64-pixels.

## Architcure
Overall Architecture

![image](https://github.com/TomGoesGitHub/Human-Faces-with-DCGANs/assets/81027049/b8acbc20-116f-4f6a-a98f-ea41ce059004)
\
\
\
CNN-Architecture of the Generator (The descriminator is modelled symmetricly)

![image](https://github.com/TomGoesGitHub/Human-Faces-with-DCGANs/assets/81027049/37c18c5e-5003-44e6-9f79-c76a3980ae18)



## GAN Best-Practices
To guarantee a stable training and desired model properties several best practices have been established, which were implemented in this project.
- One Sided Label Smoothing replaces the y=1 targets for real samples with smoothed values, like y\=0.9. Thereby, the vulnerability of neural networks to adversarial examples is reduced.
- Feature Matching: Instead of directly maximizing the output of the discriminator, the new objective encourages the generator to generate data that matches the statistics of the real data. Specifically, we train the
  generator to match the expected value of the features on an intermediate layer of the discriminator (deviations are penalized based on their squared l2-norm).
- Historical Averaging modifies each player’s cost to include a term that penalizes deviations of the learnable parameters from their historical average
- Virtual Batch Normalization addresses the problem that classic batch normalization (which has been shown to greatly improves optimization of neural networks) causes the output of a neural network for an input example to be highly
  dependent on several other inputs in the same minibatch. To avoid this problem a single reference batch of examples is chosen once and fixed at the start of training and then used for normalization throughout the training.
- Pooling Layers should be replaced by strided convolutions. Batch Normalization should be applied in both the generator and the discriminator. Fully connected hidden layers should be removed. ReLU activation should be used
  in the generator for all layers except for the output, which uses Tanh. (Note, that this requires to rescale the images into the [-1,1]interval in a preprocessing step.) LeakyReLU activation should be used in the discriminator
  for all layers except the output, which uses Sigmoid. As Gradient Optimizer Adam is suggested.

## Learning Curves
![image](https://github.com/TomGoesGitHub/Human-Faces-with-DCGANs/assets/81027049/2842836b-00b7-49f6-bc6b-59a3a616865c)


