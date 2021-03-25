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
