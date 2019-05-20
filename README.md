# Studying convective organisation with neural networks

This repository contains code to generate training data, train a neural network
model and interprete this model. 

## Getting started

### Project layout

To keep things organised I use the following layout for the directory structure
for my project:

```
.
├── code
│   └── convml_tt  <-- `convml_tt` directory from this repository symlinked here
│       ├── architectures
│       ├── data
│       │   └── satdata  <-- the `satdata` repository is cloned into here
│       └── interpretation
├── data
│   └── storage -> /nfs/see-fs-02_users/earlcd/datastore/a321/ml-data/
├── notebooks
│   ├── model_interpretation
│   └── model_training
├── papers
│   └── clouds2vec
│       ├── figures
│       └── sections
...
```

I do this to avoid checking data into the github repository while storing this
data on the a separate NFS mount.

To achieve this you could check out this repo where you store your github
repositories

```bash
cd ~

mkdir git-repos
cd git-repos
git clone https://github.com/leifdenby/convml_tt

mkdir ~/ml_project
mkdir ~/ml_project/code
mkdir ~/ml_project/data
mkdir ~/ml_project/notebooks

ln -s ~/git-repos/convml_tt/convml_tt ~/ml_project/code/convml_tt
```

### Installing dependecies

To train the model and do the model interpretation you will need to install the
fastai and pytorch python. At the moment I am using fastai `v1.0.48`. It's
easiest to install and keep track of the python modules using
[conda](https://www.anaconda.com/distribution/). Once conda is installed create
a new conda environment and install the dependencies:

```bash
conda create -n fastai python=3.6 -y
conda activate fastai
conda install matplotlib xarray netCDF4 jupyter
pip install fastai==1.0.48
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

## Training

### Training data

A small example training dataset (2000 triplets for training and 500 for study)
can be download
[here](https://leeds365-my.sharepoint.com/:u:/g/personal/earlcd_leeds_ac_uk/Ee-_nQExD9VCpWYEj8oBA1UBQ0XA6X8GlrXNdBVNe06jQg?e=s1cEBY).
**NOTE**: it's 1.10GB in size(!) To use the directory structure above this
tar-ball should be extracted into `~/ml_project/data/storage/tiles/goes16/`

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
