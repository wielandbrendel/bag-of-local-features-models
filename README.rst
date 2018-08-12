=======
BagNets
=======

In this repository you find the model specification and pretrained weights for the bag-of-local-feature models published in

`Approximating CNNs with Bag-of-local-Features models8works surprisingly well on ImageNet <http://arxiv.org/abs/XXXX.XXXX>`__.
Wieland Brendel and Matthias Bethge, arXiv:XXXX.XXXX

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git


Usage
-----

The code provides simple means to initialize the models in either Pytorch or Keras. After installation please use the following
code snippets to load the models:

.. code-block:: python

   import bagnets.pytorch
   pytorch_model = bagnets.pytorch.bagnet16(pretrained=True)

.. code-block:: python

   import bagnets.keras
   keras_model = bagnets.keras.bagnet16(pretrained=True)

and replace bagnet16 with whatever size you want (available are bagnet8, bagnet16 and bagnet32). The last number refers to the
maximum local patch size that the network can integrate over.


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
    title={Approximating CNNs with Bag-of-local-Features models8works surprisingly well on ImageNet},
    author={Brendel, Wieland and Bethge, Matthias},
    journal={arXiv preprint arXiv:XXXX.XXXX},
    year={2018},
    url={http://arxiv.org/abs/XXXX.XXXX},
    archivePrefix={arXiv},
    eprint={XXXX.XXXX},
  }

You can find the paper on arXiv: https://arxiv.org/abs/XXXX.XXXX

Authors
-------

* `Wieland Brendel <https://github.com/wielandbrendel>`_