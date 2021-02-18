# Studying convective organisation with neural networks

This repository contains code to generate training data, train and interprete
the neural network used in [L. Denby
(2020)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085190)
collected in a python module called `convml_tt`.

## Getting started

The easiest way to work with `convml_tt` is to set up a conda environment with
the necessary dependencies and then install `convml_tt` into this environment
with pip. If you are planning on modifying `convml_tt` itself have a look in
[README.dev.md](README.dev.md) for some suggestions about how to organise your
workflow.

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

For GPU-based training (with CUDA `v9`):

```bash
conda env create -f environment-gpu-cuda9.yml
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

Now you will have `convml_tt` available whenever you activate the
`convml_tt` conda environment. You will have the *base* components of
`convml_tt` installed which enable training the model on a existing
triplet-dataset and making predictions with a trained model. To produce
training data for `convml_tt` more dependecies are required depending on
the kind of input data you want to use (see "Creating training data"
below).


## Training

### Training data

#### Example dataset

A small example training dataset (2000 triplets for training and 500 for study)
can be download
[here](https://leeds365-my.sharepoint.com/:u:/g/personal/earlcd_leeds_ac_uk/Ee-_nQExD9VCpWYEj8oBA1UBQ0XA6X8GlrXNdBVNe06jQg?e=s1cEBY).
**NOTE**: it's 1.10GB in size(!) To use the directory structure above this
tar-ball should be extracted into `~/ml_project/data/storage/tiles/goes16/`


#### Creating training data from GOES-16 satellite observations

To work with satellite data you will need packages that kind read this
data, reproject it and plot it on maps. These requires some system
libraries that can be difficult to install using only `pip`, but can
easily be installed with conda into your `convml_tt` environment

```bash
conda install -c conda-forge xesmf cartopy
```

And then use pip to install the matching python packages

```bash
pip install .[sattiles]
```

**TODO**: complete rest of guide talking about processing pipeline and
downloading satellite data

An example of how to generate a dataset can be see in
`training_gen_examples/goes16_training_and_study.py`. This script attempts to
store training and study triplets in `data/storage/triplets/` and attempts to
read from `data/storage/sources/goes16` to create composites for the analysis
domain. The example will fetch GOES-16 data from Amazon S3 if it isn't found
locally.


### Training on ARC3

First log in to ARC3

```bash
[earlcd@cloud9 ~]$ ssh arc3.leeds.ac.uk
```

Request a node with a GPU attached

```bash
[earlcd@login1.arc3 ~]$ qrsh -l coproc_k80=1,h_rt=1:0:0 -pty y /bin/bash -i
```

Once the node has spun up and the prompt is again available you will notice the
hostname has now changed (to something containing "gpu", here `db12gpu1`). Now
activate your conda environment (installed as above) and start a jupyter
notebook

```bash
[earlcd@db12gpu1.arc3 ~]$ conda activate fastai
[earlcd@db12gpu1.arc3 ~]$ jupyter notebook --no-browser --port 8888 --ip=0.0.0.0
```

Finally from your workstation start a second ssh connection to ARC3, this time
forwarding the local port `8888` to port `8888` on the GPU node (here
`db12gpu1`) you have been allocated on ARC3:

```bash
[earlcd@cloud9 ~]$ ssh arc3 -L 8888:db12gpu1.arc3.leeds.ac.uk:8888
```

Open up a local browser on your workstation and browse to
`http://localhost:8888`

### Training example

There is an example jupyter notebook of how to load the training data and train
the model in `example_notebooks/model_training`. The notebook expects the
directory layout mentioned above, but you can just modify it for your own
needs.

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
