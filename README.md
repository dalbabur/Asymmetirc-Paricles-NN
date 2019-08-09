

# Asymmetric Particles

The aim of this project is to study the **dynamic behavior of asymmetric particles** under **inertial focusing** conditions (high Re) in microchannels. Properties such as focusing position/s, angular velocity, and preferred orientation, could shine a light on the forces that govern this phenomenon, and inspire novel bioassays based on asymmetric particles.

In order to extract these properties from the raw footage, it is necessary to **detect** and **track** the particles in every frame, as well as to find their **orientation**. Because traditional image processing methods lacked the necessary robustness to carry out a thorough analysis (errors on particle detetectoin / center of mass, and orientation), we implemented **neural networks** to execute these tasks. Ideally, the two main models will produce first a **binary mask** encoding the pixels of each frame which belong to the particle of interest as opposed to the background (from which we can easily calculate the center of mass), and then the corresponding **angle of rotation** (orientation) for each particle.

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
	 Make sure you familiriaze yourself with the git workflow.

 - **About this repo**:
	  structure of repo


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
*Disclaimer*: this is all for Windows 10, macOS or Linux could be very different.

### Prerequisites
Things you need before using this repo and how to install them. I recommned reading through all of the links before actually installing anything.

 - **Language**: Python - [Anaconda](https://www.anaconda.com/distribution/)
	(would also recommned using virtual environments from here on, easy enough to set up with the Anaconda GUI or cmd)

 - **ML Engine**: TensorFlow - [CPU](https://www.tensorflow.org/install/) or [GPU](https://www.tensorflow.org/install/gpu)  
 (if GPU, also install NVIDIA [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/))

 - **High Level NN Wrapper**: [Keras](https://keras.io/#installation)

 - **Your IDE of choice**: PyCharm, Atom, Visual Studio... I enjoy [Atom](https://atom.io) + [Hydrogen](https://nteract.gitbooks.io/hydrogen/)

 - **Other Packages**: some standard python packages and their corresponding version are listed in **requirements.txt**

 - You should also have **MATLAB** (or [port everthing to python, more on this later](#last-thoughts-current-state-and-future-direction))

 - **Git**: I recommend [GitHub Desktop](https://desktop.github.com) as it will install Git for you, and has a nice GUI but it is not necessary. If not, you'll need to install [Git](https://git-scm.com/downloads) yourself, and/or other GUIs.

### Installing

Once everything is installed, you can just download this repo onto your machine, and you'll be able to commit and push updates. You will also need the **data from Drobo**, and the trained weights if you don't want to train the models yourslef (altough you will eventually have to). **Keep the repo structure the same** to make the next step easier.

The first thing you'll need to do is **check all the directories** and make the necessary changes. Make sure all variables in the following scripts are poiting to the right folder/data.

 - *code\backbone\predict.py* (most important)
 - *code\ClassNet\train_classification.py* (only if you're training)
 - *code\RotNet\train_classification.py* (only if you're training)
 - *code\UNET\train_segmentation.py* (only if you're training)
 - *code\UNET\visualize.py* (only if you're training: should be decapricated)
 - *code\UNET\generate_fake.m* (only if you're generating segmentation training data)

The main code you need to execute if you have trained models is *backbone\predict.py*. It is **best to run line-by-line** (in the style of [Jupyter](https://jupyter.org) or [Hydrogen](https://nteract.gitbooks.io/hydrogen/)) and check the ouputs as you go.

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
 - using *code\cine_utils\spitObj.m*: hijacked code from Chris that will correlate each frame to a background (frame with no particles) and save those frames with lower correlation than a set threshold. Pros: automatic. Cons: empirical (threshold), takes long time to run, unaccurate.
 - exploiting Annie's work: there are .mat files for every movie that contain information about her analysis. There is a variable that containts all frame numbers with particles: it's only a matter of extracting those frame numbers are reading them. Pros: much faster. Cons: unsure of Annie's accuracy, not yet implmented. [More on this later.](#last-thoughts-current-state-and-future-direction)

This raw data needs to be transfomed into the correct input for each neural network, a transoformation that will be addressed in the corresponding NN section.


### Image segmentation
Since the raw data is not suited to trained a neural networks (it's not labeled), there are some options:

 - Label the data with tools along the lines of [Labelbox](https://labelbox.com)
 - Use unsupervised learning, like [this](https://arxiv.org/abs/1711.08506) or [this](https://link.springer.com/article/10.1007/s11263-019-01183-3)
 - Make training data that resembles the raw data, but produce the binary mask at the same time (fake or synthetic data)
	 - Extract empty frames and crop images of particles, and the collage them together at random, adding small transformations and noise. This assumes we can generate the statistical distribution of our raw data from a few samples.



### Particle classification

Explain what these tests test and [why](#last-thoughts-current-state-and-future-direction)

### Particle orientation

Explain what these tests test and [why](#last-thoughts-current-state-and-future-direction)


### Backbone: putting everything together

Explain what these tests test and [why](#last-thoughts-current-state-and-future-direction)


## Last Thoughts: Current State and Future Direction

## Authors

* **Diego Alba** - *Initial work* - [GitHub](https://github.com/DIEGOA363) [LinkedIn](https://www.linkedin.com/in/dalbabu/)
