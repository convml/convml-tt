# Notes for using ARC3

## Installing conda on ARC3

ARC3 doesn't have conda installed so we first need to install conda. **NOTE**:
replace `earlcd` with your own username in the instructions below.

The commands below will create a directory for yourself in `/nobackup` on ARC3
and install conda there

```bash
mkdir -p /nobackup/$USER/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /nobackup/$USER/miniconda3
source ~/.bashrc
```

You should now have `conda` available and can install the dependencies.

## Setting up SSH aliases and keys

Defining aliases and settings up ssh keys will save you a lot of time being
able to directly ssh into ARC3 and copy to/from ARC3 in one command.

1. In the following replace `earlcd` with your own username and paste into
   `.ssh/config` (create the file if it doesn't exist):

```
Host leeds-login
        HostName feeble.leeds.ac.uk
        User earlcd

Host leeds-arc3
        HostName arc3.leeds.ac.uk
        ProxyCommand ssh leeds-login -Y -W %h:%p
        User earlcd
```

2. Create an ssh-key

```
# ssh-keygen
```

3. Use `ssh-copy-id` to set up passwordless login to `leeds-login` and
   `leeds-arc3`. After each of these commands you will be prompted for your
   password.

```bash
ssh-copy-id leeds-login
ssh-copy-id leeds-arc3
```

You can now directly ssh into ARC3 with

```
ssh leeds-arc3
```

And copy files with for example

```
scp localfile.txt leeds-arc3:~/{path-in-homedir}/
```


## Training on ARC3

First log in to ARC3

```bash
[earlcd@cloud9 ~]$ ssh arc3.leeds.ac.uk
```

You can either a) run the training interactively by first requesting an
interactive job from the scheduler with a GPU attached or b) submit a batch job
to the scheduler to do the training non-interactively

### a) Interactive job

While you're making sure that you have everything installed correctly and the
training data in the correct place it can be easiest to do the training
interactively.

Request a node with a GPU attached

```bash
[earlcd@login1.arc3 ~]$ qrsh -l coproc_k80=1,h_rt=1:0:0 -pty y /bin/bash -i
```

Once the node has spun up and the prompt is again available you will notice the
hostname has now changed (to something containing "gpu", here `db12gpu1`). Now
activate your conda environment (installed as above) and start the training

```bash
[earlcd@db12gpu1.arc3 ~]$ conda activate convml_tt
[earlcd@db12gpu1.arc3 ~]$ python -m convml_tt.trainer <path-to-your-dataset>
```

Or you can open a jupyter notebook and forward traffic from the GPU so that you
can run the training using a jupyter notebook:

```bash
[earlcd@db12gpu1.arc3 ~]$ jupyter notebook --no-browser --port 8888 --ip=0.0.0.0
```

To do this you'll need to start a second ssh connection to ARC3 from your work
station, this time forwarding the local port `8888` to port `8888` on the GPU
node (here `db12gpu1`) you have been allocated on ARC3:

```bash
[earlcd@cloud9 ~]$ ssh arc3 -L 8888:db12gpu1.arc3.leeds.ac.uk:8888
```

Open up a local browser on your workstation and browse to
`http://localhost:8888`

### b) Batch job training

To submit a job to the SGE scheduler on ARC3 you will need a script like the following:

```bash
#!/bin/bash

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

# request one hour
#$ -l h_rt=01:00:00

# with on k80 GPU
#$ -l coproc_k80=1

# setup conda, change this to the path where you have conda installed
. "/nobackup/earlcd/anaconda3/etc/profile.d/conda.sh"
conda activate convml_tt

# extract data to TMPDIR
echo "Extract data to $TMPDIR"
DATAFILE="/nobackup/earlcd/convml_tt/data/goes16__Nx256_s200000.0_N500study_N2000train.tar.gz"
tar zxf $DATAFILE -C $TMPDIR
ls $TMPDIR
tree -d $TMPDIR

# run training
DATA_PATH="$TMPDIR/Nx256_s200000.0_N500study_N2000train/"
python -m convml_tt.trainer $DATA_PATH --gpus 1 --num-dataloader-workers 24 --max-epochs 10 --log-to-wandb --batch-size 128
```

The above script ensures you have a node with a GPU, have conda available and
the environment activated and also copies the dataset to the local storage on
the node where you are training (which is a lot faster than using `nobackup` to
read training data from).
