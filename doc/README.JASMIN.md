# Training convml_tt TripletTrainerModel on JASMIN

Below is an example submission script for training on the
[JASMIN](https://www.jasmin.ac.uk/) data analysis facility. This assumes that
`convml_tt` and its requirements have already been installed into a conda
environment called `convml_tt`.

```bash
#!/bin/bash 
#SBATCH --partition=lotus_gpu
#SBATCH --account=lotus_gpu
#SBATCH -o convml_tt.%a.%j.out 
#SBATCH -e convml_tt.%a.%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -n 32

# make sure we have 32 cores so that we get access to all the memory on the node

DATASET="Nx256_s200000.0_N500study_N2000train"

# make sure conda is available
source /home/users/lcdenby/.bashrc
# load conda env
conda activate convml_tt
# and train
python -m convml_tt.trainer $DATASET --gpus 1 --max-epochs 100 --log-to-wandb --preload-data --num-dataloader-workers 32 --batch-size 64
```
