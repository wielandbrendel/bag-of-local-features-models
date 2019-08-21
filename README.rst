=======
BagNets
=======

In this repository you find the model specification and pretrained weights for the bag-of-local-feature models published in

| `Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet <https://openreview.net/pdf?id=SkfMWhAqYQ>`__.
| Wieland Brendel and Matthias Bethge, ICLR 2019

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git


Usage
-----

The code provides simple means to initialize the models in either Pytorch or Keras. After installation please use the following
code snippets to load the models:

.. code-block:: python

   import bagnets.pytorchnet
   pytorch_model = bagnets.pytorchnet.bagnet17(pretrained=True)

.. code-block:: python

   import bagnets.kerasnet
   keras_model = bagnets.kerasnet.bagnet17()

and replace bagnet17 with whatever size you want (available are bagnet9, bagnet17 and bagnet33). The last number refers to the
maximum local patch size that the network can integrate over.

FAQ
----

*   **Do I need to manually split the image into patches?**

    No. You use BagNets just like any other DNN and apply it to the whole image. The BagNets are really similar to ResNets. In a nutshell we simply replaced most 3x3 convolutions by 1x1 convolutions. This effectively means that the largest receptive fields in the BagNets are of size qxq (where q is smaller than the image size), which is equivalent to splitting the image into individual patches.

Image Preprocessing
-------------------

The models expect inputs with the standard torchvision preprocessing, i.e.

* with RGB channels
* in the format [channel, x, y]
* loaded with pixel values between 0 and 1 which are then...
* ...normalized by mean and standard deviation, i.e. for given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, the normalization should transform each channel of the input as input[channel] = (input[channel] - mean[channel]) / std[channel]

The mean and standard deviation are:

* mean = [0.485, 0.456, 0.406]
* std = [0.229, 0.224, 0.225]

Citation
--------

If you find BagNets useful for your scientific work, please consider citing it
in resulting publications:

.. code-block::

  @article{brendel2018bagnets,
    title={Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet},
    author={Brendel, Wieland and Bethge, Matthias},
    journal={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/pdf?id=SkfMWhAqYQ},
  }

You can find the paper on OpenReview: https://openreview.net/pdf?id=SkfMWhAqYQ

Authors
-------

* `Wieland Brendel <https://github.com/wielandbrendel>`_
