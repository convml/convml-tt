# Studying convective organisation with neural networks

This repository contains code to generate training data, train and interprete
the neural network used in [L. Denby
(2020)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085190)
collected in a python module called `convml_tt`. From version `v0.7.0` it
was rewritten to use [pytorch-lightning](https://pytorchlightning.ai/) rather
than [fastai v1](https://fastai1.fast.ai/) to adopt best-practices and make it
easier to modify and carry out further research on the technique.

## Getting started

The easiest way to work with `convml_tt` is to set up a conda environment with
the necessary dependencies and then install `convml_tt` into this environment
with pip.

**1. Check out `convml_tt` from github**

```bash
git clone https://github.com/leifdenby/convml_tt
cd convml_tt
```

**2. Install dependecies**

To train the model and do the model interpretation there are number of python
modules which are needed. All the necessary dependencies can be installed with
[conda](https://www.anaconda.com/distribution/). Once conda is installed you
can create an environment depending on whether you will be doing GPU or
CPU-based training

For GPU-based training:

```bash
conda env create -f environment-gpu.yml
conda activate convml_tt
```

For CPU-based training:

```bash
conda env create -f environment-cpu.yml
conda activate convml_tt
```

**3. Install `convml_tt`**

Once you have a conda environment set up and **activated** you can install
`convml_tt` through pip with:

```bash
pip install .
```

You will now have `convml_tt` available whenever you activate the `convml_tt`
conda environment. You will have the *base* components of `convml_tt`
installed which enable training the model on a existing triplet-dataset
and making predictions with a trained model. To produce training data for
`convml_tt` more dependecies are required depending on the kind of input
data you want to use (see "Creating training data" below).

**NOTE ON DEVELOPING `convml_tt`**: if you plan on modifying the `convml_tt`
code yourself you add the `-e` flag above (i.e. use `pip install -e .`) so that
any changes you make are automatically picked up.


## Training

Below are details on how to obtain training data and how to train the model

### Training data

#### Example dataset

A few example training datasets can be downloaded using the following
command

```bash
python -m convml_tt.data.examples
```


#### Creating training data from GOES-16 satellite observations

**NB**: dataset creation doesn't currently work as it is being refactored

To work with satellite data you will need packages that kind read this
data, reproject it and plot it on maps. These requires some system
libraries that can be difficult to install using only `pip`, but can
easily be installed with conda into your `convml_tt` environment

```bash
conda install -c conda-forge xesmf cartopy
```

And then use pip to install the matching python packages

```bash
pip install ".[sattiles]"
```

**TODO**: complete rest of guide talking about processing pipeline and
downloading satellite data

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

Finally detailed notes on how to train on the ARC3 HPC cluster at University of
Leeds are in [doc/README.ARC3.md](doc/README.ARC3.md).

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
