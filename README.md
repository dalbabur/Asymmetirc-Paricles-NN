# Asymmetric Particles

The aim of this project is to study the **dynamic behavior of asymmetric particles** under **inertial focusing** conditions (high Re) in microchannels. Properties such as focusing position/s, angular velocity, and preferred orientation, could shine a light on the forces that govern this phenomenon, and inspire novel bioassays based on asymmetric particles.

In order to extract these properties from the raw footage, it is necessary to **detect** and **track** the particles in every frame, as well as to find their **orientation**. Because traditional image processing methods lacked the necessary robustness to carry out a thorough analysis (errors on particle detection / center of mass, and orientation), we implemented **neural networks** to execute these tasks. Ideally, the two main models will produce first a **binary mask** encoding the pixels of each frame which belong to the particle of interest as opposed to the background (from which we can easily calculate the center of mass), and then the corresponding **angle of rotation** (orientation) for each particle.

This repository contains all code pertaining to the different neural networks (.py), the data generation for training and validating the models (.m), and some example movies for the current state of the analysis (.avi). It **DOES NOT** contain the raw data (.cine), the individual frames (.tif), the cropped images (.tif), the binary masks (.tif), or the trained weights (.h5) for the networks. The copy of the repository backed up in **Drobo** does contain all of these files.

### Background Information

 - **Inertial Focusing and Previous Work**:

	Maybe some links about inertia focusing?  
	Maybe link to Annie's paper?

 - **Neural Networks**:

	Link to Neural Network in a Nutshell presentation by Diego  
	Philosophy behind synthetic data for training


	What does the data look like
	Some links about neural networks

 - **About GitHub**:
	 what
	 Make sure you familiarize yourself with the git workflow.

 - **About this repo**:
	  structure of repo


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
*Disclaimer*: this is all for Windows 10, macOS or Linux could be very different.

### Prerequisites
Things you need before using this repo and how to install them. I recommend reading through all of the links before actually installing anything.

 - **Language**: Python - [Anaconda](https://www.anaconda.com/distribution/)
	(would also recommend using virtual environments from here on, easy enough to set up with the Anaconda GUI or cmd)

 - **ML Engine**: TensorFlow - [CPU](https://www.tensorflow.org/install/) or [GPU](https://www.tensorflow.org/install/gpu)  
 (if GPU, also install NVIDIA [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/))

 - **High Level NN Wrapper**: [Keras](https://keras.io/#installation)

 - **Your IDE of choice**: PyCharm, Atom, Visual Studio... I enjoy [Atom](https://atom.io) + [Hydrogen](https://nteract.gitbooks.io/hydrogen/)

 - **Other Packages**: some standard python packages and their corresponding version are listed in **requirements.txt**

 - You should also have **MATLAB** (or [port everything to python, more on this later](#last-thoughts-current-state-and-future-direction))

 - **Git**: I recommend [GitHub Desktop](https://desktop.github.com) as it will install Git for you, and has a nice GUI but it is not necessary. If not, you'll need to install [Git](https://git-scm.com/downloads) yourself, and/or other GUIs.

### Installing

Once everything is installed, you can just download this repo onto your machine, and you'll be able to commit and push updates. You will also need the **data from Drobo**, and the trained weights if you don't want to train the models yourself (although you will eventually have to). **Keep the repo structure the same** to make the next step easier.

The first thing you'll need to do is **check all the directories** and make the necessary changes. Make sure all variables in the following scripts are pointing to the right folder/data.

 - *code\backbone\predict.py* (most important)
 - *code\ClassNet\train_classification.py* (only if you're training)
 - *code\RotNet\train_classification.py* (only if you're training)
 - *code\UNET\train_segmentation.py* (only if you're training)
 - *code\UNET\visualize.py* (only if you're training; should be decapricated ([more on this later](#last-thoughts-current-state-and-future-direction)))
 - *code\UNET\generate_fake.m* (only if you're generating segmentation training data)

The main code you need to execute if you have trained models is *backbone\predict.py*. It is **best to run line-by-line** (in the style of [Jupyter](https://jupyter.org) or [Hydrogen](https://nteract.gitbooks.io/hydrogen/)) and check the outputs as you go.

## Unravelling the Code

This section, as well as the code, is broken up into the following topics:

 1. [The raw data and the training data](#the-raw-data-and-the-training-data)
 2. [Image segmentation](#image-segmentation)
 3. [Particle classification](#particle-segmentation)
 4. [Particle orientation](#particle-orientation)
 5. [Backbone: putting everything together](#backbone-putting-everything-together)

### The raw data and the training data

The raw data (.cine) can be found in Chris' Drobo; the folder/file name indicate which particle and which flowrate. The particles are relatively sparse (most frames contain no particles), so you'll need to spend some time extracting the relevant frames and converting them to .tif images. Some ways of doing this:

 - by hand: in PCC look at each frame and save as .tif if contains a particle. Pros: accurate. Cons: time consuming, mindless.
 - using *code\cine_utils\spitObj.m*: hijacked code from Chris that will correlate each frame to a background (frame with no particles) and save those frames with lower correlation than a set threshold. Pros: automatic. Cons: empirical (threshold), takes long time to run, inaccurate.
 - exploiting Annie's work: there are .mat files for every movie that contain information about her analysis. There is a variable that contains all frame numbers with particles: it's only a matter of extracting those frame numbers are reading them. Pros: much faster. Cons: unsure of Annie's accuracy, not yet implemented. [More on this later.](#last-thoughts-current-state-and-future-direction)

This raw data needs to be transformed into the correct input for each neural network, a transformation that will be addressed in the corresponding NN section.


### Image segmentation
Everything pertaining to image segmentation should be in [*code\UNET*](https://github.com/ijungsj/Asymmetric-Particles/tree/master/code/UNET) (data folder, weights folder, and several scripts)

***The Data***:  
Since the raw data is not suited to trained a neural networks (it's not labeled), there are some options:

 - Label the data with tools along the lines of [Labelbox](https://labelbox.com)
 - Use unsupervised learning, like [this](https://arxiv.org/abs/1711.08506) or [this](https://link.springer.com/article/10.1007/s11263-019-01183-3)
 - Make training data that resembles the raw data, but produce the binary mask at the same time (fake or synthetic data)
	 - Extract empty frames and crop images of particles, and the collage them together at random, adding small transformations and noise. This assumes we can generate the statistical distribution of our raw data from a few samples.
	 - When the particle images are cropped, it is important to erase the background and save it as a .png to ensure we can make the binary masks later. You can use [Photopea](https://www.photopea.com) or any photo editor for this.

 As of right now, the repo uses the third option. To generate the training and validation images you'll need:

 - *code\UNET\data\synthetic*, which contains background images and cropped particles
 - *code\UNET\generate_fake.m*, which needs the following parameters:

``` matlab
generate = 5120; 		% how many images to generate
max_objs = 7; 			% maximum number of objects per frame
folder = '\data\train'; % folder to save the generated images to, either \train or \test
noise = 1; 				% boolean, whether or not to add poisson noise to the particle
transform = 1; 			% boolean, whether or not to transform the particles (rotate,resize,shear,crop)

path = '...\AsymParticles\code\UNET'; 			% full path to \UNET
bgpath = [path,'\data\synthetic\background\']; 	% path for the background images

% add PARTICLEpath for each particle
Lpath = [path,'\data\synthetic\L\object\']; 	% path to the cropped L images
Upath = [path,'\data\synthetic\U\object\']; 	% path to the cropped U images

% list all images, add them to a cell array
bg = dir([bgpath '*.tif']);
U = dir([Upath '*.png']);
L = dir([Lpath '*.png']);
dirs = {U,L};
```
This will generate 5120 images with a number of 0 to 7 particles, which can either be Us or Ls, after applying random transformations to it. It will also generate the corresponding 5120 masks, where 0 is background and 1 and 2 are the U and L particles, respectively.

It's easy to add more particles (+, x, s...) by cropping them and making new folders with the same structure.

***The Model***:  
Defined in [code\UNET\UNET_model.py](https://github.com/ijungsj/Asymmetric-Particles/tree/master/code/UNET/UNET_model.py), based on the [UNET architecture](https://arxiv.org/pdf/1505.04597.pdf). This file also defines other loss functions (mainly used for multiclass segmentation), such as the [Tversky loss](https://arxiv.org/abs/1706.05721). For binary segmentation, binary crossentropy works fine.

***The Training***:  
Defined in [code\UNET\train_segmentation.py](https://github.com/ijungsj/Asymmetric-Particles/tree/master/code/UNET/train_segmentation.py). Mainly follows the Keras workflow (below, 1-5) and needs the following parameters:
```python
data_size = 5170 		# total number of frames for training  
test_size = 2560 		# total number of frames for testing
classes = 2 			# number of classes to segment
BATCH_SIZE = 5			# number of frames to take at once (mainly depends on your memory size)
epoch = 20  			# number of times to loop through the entire data
resize = (64,640)					# new dimensions for the frame (raw is 128x1280)
weights = 'code/UNET/UNET_bin.h5' 	# where to save the weights after training is complete
```
 1. make ImageDataGenerator/s (rescale the grayscale images to 0-1, make sure mask are binary (or not, if doing multiclass))
 2. call flow_from_directory (point to data folder, resize, do one hot-encoding)
 3. define callbacks
 4. make UNET model
 5. call fit_generator (actually train the model)

You can then visualize the training curve, see the output images, and the activations for each layer. Check the end of the script as well as [code\UNET\visualize.py](https://github.com/ijungsj/Asymmetric-Particles/tree/master/code/UNET/visualize.py)

Beware, I modified the original Keras files (mainly ImageDataGenerator and flow_from_directory) to add aditional functionality. Everything is self-contained in [code\UNET\image.py](https://github.com/ijungsj/Asymmetric-Particles/tree/master/code/UNET/image.py), although very lengthy and perhaps messy. I would recommend looking at the Keras documentation and source files first before digging into it.

### Particle classification

Explain what these tests test and [why](#last-thoughts-current-state-and-future-direction)

### Particle orientation

Explain what these tests test and [why](#last-thoughts-current-state-and-future-direction)


### Backbone: putting everything together

Explain what these tests test and [why](#last-thoughts-current-state-and-future-direction)


## Last Thoughts: Current State and Future Direction

## Authors

* **Diego Alba** - *Initial work* - [GitHub](https://github.com/DIEGOA363) [LinkedIn](https://www.linkedin.com/in/dalbabu/)
