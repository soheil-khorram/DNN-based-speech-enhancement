.. -*- mode: rst -*-

DNN-based-speech-enhancement
============================

.. image:: Fig2.png

What is it?
-----------

This repository contains a python implementation of a deep neural network (DNN)-based speech enhancement system. It uses Keras with tensorflow back-end to train and test neural networks. The details of the implemented speech enhancement system are explained in the paper [1]. The system supports for the following features:

   (1) Standard and dilated convolutional networks.

   (2) Causal and non-causal convolutional networks.

   (3) Three different styles of enhancement: vanilla, spectral-subtraction and Wiener.
   
Vanilla CNN directly generates the enhanced signal from the input noisy signal; spectral-subtraction-style CNN first predicts noise and then generates the enhanced signal by subtracting noise from the noisy signal; Wiener-style CNN generates an optimal mask for suppressing noise. 

How to run it?
--------------

To repeat the experiments of the paper [1], you can run the "run.sh" file with two arguments as:

.. code-block:: bash

   ./run.sh dataset_directory output_directory

Before running this line, make sure that your dataset files are in the dataset_directory folder and they are in a correct format. Samples should be stored in .mat format. The dataset_directory must contain the following files: "noisy1.mat", "clean1.mat",
"noisy2.mat", "clean2.mat", ..., "noisyN.mat", "cleanN.mat", where N is the total number of samples; "noisyi.mat" and "cleani.mat" are the noisy and clean versions of the i-th sample. Each mat file stores a D-by-T matrix, where D is the dimensionality of the features and T is the total number of frames. In all experiments of the paper [1], all samples are zero padded to a fixed number of frames (T = 2500). To change the format of the files, you need to change the load function in the DataLoader class.

References
----------

.. [1] Nursadul M., Khorram S., Hansen J.,
       *"Convolutional Neural Network-based Speech Enhancement for Cochlear Implant Recipients"*,
       Interspeech, 2019. [`PDF <https://arxiv.org/pdf/1907.02526.pdf>`_]

Author
------

- Soheil Khorram, 2019

Please contact khorram.soheil@gmail.com, if you have any question regarding this repository.
