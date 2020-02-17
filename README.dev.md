# Making changes to `convml_tt`

When working on `convml_tt` I add the source path for `convml_tt` directly to
the `PYTHONPATH` environment variable instead of installing `convml_tt` through
pip. Below are some notes for how to get started.

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
