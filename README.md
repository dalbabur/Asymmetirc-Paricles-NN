
# Asymmetric Particles

The aim of this project is to study the **dynamic behavior of asymmetric particles** under **inertia focusing** conditions (high Re) in microchannels. Properties such as focusing position/s, angular velocity, and preferred orientation, could shine a light on the forces that govern this phenomenon, and inspire novel bioassays based on these particles.

In order to extract these properties from the raw footage, it is necessary to **detect** and **track** the particles in every frame, as well as to find their **orientation**. We use neural networks.......

This repository contains all code pertaining to the different neural networks (.py), the data generation for training and validating the models (.m), and some example movies for the current state of the analysis (.avi). It **DOES NOT** contain the raw data (.cine), the individual frames (.tif), the cropped images (.tif), the binary masks (.tif), or the trained weights for the networks. The copy of the repository backed up in **Drobo** does contain all of these files.

### Background Information

What does the data look like
Philosophy behind synthetic data for training

Maybe some links about inertia focusing?
Maybe link to Annies paper?

Some links about neural networks

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

Python - Anaconda
keras
tensorflow --gpu
nvidia cudnn
some IDE (pycharm, atom)

Links for everything

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Unravelling the Code

Explain how to run the automated tests for this system

 1. Data
 2. Segmentation
 3. Classification
 4. Orientation
 5. Backbone

### Training/Validation Data Generation

Explain what these tests test and why

```
Give an example
```

### Neural Network

Explain what these tests test and why

```
Give an example
```

## Last Thoughts: Current State and Future Direction

## Authors

* **Diego Alba** - *Initial work* - [GitHub](https://github.com/DIEGOA363)
