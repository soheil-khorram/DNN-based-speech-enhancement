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

Contains all codes related to the paper:

References
----------

.. [1] Nursadul M., Khorram S., Hansen J.,
       *"Convolutional Neural Network-based Speech Enhancement for Cochlear Implant Recipients"*,
       Interspeech, 2019.

Author
------

- Soheil Khorram, 2019
