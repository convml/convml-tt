# Studying convective organisation with neural networks

This repository contains code to generate training data, train and interprete
the neural network used in [L. Denby
(2020)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085190)
collected in a python module called `convml_tt`. From version `v0.7.0` it
was rewritten to use [pytorch-lightning](https://pytorchlightning.ai/) rather
than [fastai v1](https://fastai1.fast.ai/) to adopt best-practices and make it
easier to modify and carry out further research on the technique.

## Getting started

To use the `convml_tt` codebase you will first need to install pytorch which
can most easily by done with [conda](https://www.anaconda.com/distribution/).

1. Once conda is installed you can create a conda environment:

```bash
conda env create -n convml-tt
conda activate convml-tt
```

Into this conda environment you the need to install pytorch. Depending on
whether you have access to a GPU or not you will need to install different
pytorch packages:

2a. For GPU-based trained and inference:

```bash
conda install pytorch torchvision>=0.4.0
```

2b. For CPU-based training and inference:

```bash
conda install pytorch torchvision>=0.4.0
```

3. With the environment set up and pytorch installed you can now install
   `convml-tt` directly from [pypi](https://pypi.org/) using pip (note if you
   are planning on modifying the `convml-tt` functionality you will want to
   download the `convml-tt` source code and install from a local copy instead
   of from pypi. See [development instructions]() for more details):

```bash
python -m pip install convml-tt
```

You will now have `convml-tt` available whenever you activate the `convml-tt`
conda environment. You will have the *base* components of `convml-tt`
installed which enable training the model on a existing triplet-dataset
and making predictions with a trained model. Functionality to create training
data is contained in a separate package called
[convml-data](https://github.com/convml/convml-data)


## Training

Below are details on how to obtain training data and how to train the model

### Training data

### Example dataset

A few example training datasets can be downloaded using the following
command

```bash
python -m convml_tt.data.examples
```

### Model training

You can use the CLI (Command Line Interface) to train the model

```bash
python -m convml_tt.trainer data_dir
```

where `data_dir` is the path of the dataset you want to use. There are a number
of optional command flags available, for example to train with one GPU use
the training process to [weights & biases](https://wandb.ai) use
`--log-to-wandb`. For a list of all the available flags use the `-h`.

Training can also be done interactively in for example a jupyter notebook, you
can see some simple examples how what commands to use by looking at the
automated tests in [tests/](tests/).

Finally there detailed notes on how to train on the ARC3 HPC cluster at
University of Leeds are in [doc/README.ARC3.md](doc/README.ARC3.md), on the
[JASMIN](doc/README.JASMIN.md) analysis cluster and on
[Google Colab](https://colab.research.google.com/drive/18Hmik9Nacqo-29b16hgQ3XfPum1lHdCO?usp=sharing).

# Model interpretation

There are currently two types of plots that I use for interpreting the
embeddings that the model produces. These are a dendrogram with examples
plotted for each class of the leaf nodes of the dendrogram and a scatter plot
of two dimensions annotated with example tiles so the actual tiles can be
visualised.

There is an example of how to make these plots and how to easily generate an
embedding (or encoding) vector for each example tile in
`example_notebooks/model_interpretation`. Again this notebook expects the
directory layout mentioned above.
