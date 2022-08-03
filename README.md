# Implementation for Task2Sim

## Installation

For installing all python packages required, run `pip install -r requirements.txt`.
Additionally, for generating data using Three-D-World(TDW), you might need to follow 
additional instructions based on your use-case following [this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/getting_started.md).

*Downloading Downstream Task Data*: Please follow the instructions [here](https://github.com/asrafulashiq/transfer_broad#download-datasets) 
for downloading data for the seen set of tasks. For unseen tasks, please follow the citations in the paper.

As a general rule, all commands below should be run from the base directory of this repository.


## Generating Three-D-World Data

The file `generator/generator.py` contains the definition of a `Generator` class, 
an object of which is the main interface for generating Three-D-World data. The `GenParams`
class in `generator/generator_params.py` defines the different simulation parameters, and 
an object of `Generator` uses and object of `GenParams` to generate a synthetic image set.
Each `GenParams` object has corresponding tuple and string representations, which are just 
a concatenation of all its parameters with a '_' separator in case of a string. 

For generating the 256 sets of size 40k images each, run
`python generator/generate_256_sets.py --root_dir /path/to/output/dir --num_nodes=1`. 
The options `num_nodes` and `idx` (where `idx` specifies node index), can be used if generation 
of the 256 datasets is being split across multiple nodes. 

Note that currently, the script assumes that the TDW build is launched manually, if not the 
`launch_build` argument in defining a `Generator` needs to be set to `True`.

## Pre-training

NOTE: Our scripts used slurm for launching jobs and for it we used the `submitit` python utility. 
All jobs used for pre-training and downstream evaluation can be appropriately modified for 
machines that do not use slurm. 
Additionally, we used `wandb` for experiment tracking and logging. If not used, it can be disabled 
using the `--debug` flag in all of the following scripts.

For pre-training a Resnet-50 backbone on an image dataset, run the command
```bash
python run_with_submitit.py --debug --ngpus 4 --cfg-yml configs/pt_default.yaml \
--cfg-override SAVE_DIR path/to/output_dir DATA_DIR path/to/pre-training/image/data
```

Equivalently, on a machine that does not use slurm for job management (and has at least 4 gpus), run
```bash
bash distributed_train.sh 4 --cfg-yml configs/pt_default.yaml \
--cfg-override SAVE_DIR path/to/output_dir DATA_DIR path/to/pre-training/image/data
```

## Downstream Evaluation

In the following, `ChestX` has been used as an example downstream task, which is 
to be replaced appropriately. Also, the hyperparameters in the override arguments
were substituted appropriately from `configs/lineval_best_hps.yaml` or 
`configs/finetune_best_hps.yaml`.

### Linear Probing
```bash
python run_with_submitit.py --debug --cfg-yml configs/lineval_defaults.yaml \
--cfg-override DOWNSTREAM_EVAL lineval DATASET ChestX \
BACKBONE_PATH path/to/pre-trained/model SAVE_DIR path/to/output_dir
LR 0.001 WD 0. BATCH_SIZE 128
```

### Full Network Fine-tuning
```bash
python run_with_submitit.py --debug --cfg-yml configs/finetune_defaults.yaml \
--cfg-override DOWNSTREAM_EVAL finetune DATASET ChestX \
BACKBONE_PATH path/to/pre-trained/model SAVE_DIR path/to/output_dir
LR 0.001 WD 1.e-5 BATCH_SIZE 128
```

### K-nearest neighbors

```bash
python run_with_submitit.py --debug --cfg-yml configs/knn_defaults.yaml \
--cfg-override DOWNSTREAM_EVAL knn DATASET ChestX \
BACKBONE_PATH path/to/pre-trained/model SAVE_DIR path/to/output_dir
```

## Training Task2Sim

Using all the above, we pre-generated 256 synthetic image datasets, pre-trained a Resnet-50 on each 
and ran downstream evaluation using a 5 nearest neighbors classifier on all 20 downstream tasks.
These accuracies, which for the seen tasks, are used as the rewards for training Task2Sim, 
are stored in `controller_db/reward_db.json`. Additionally, the Task2Vec vectors, 
which can be computed using the script `controller/task_db.py` are available in 
`task_db/resnet18_imagenet_12_seen_tasks.py` and `task_db/resnet18_imagenet_8_unseen_tasks.py`.

For training the Task2Sim model, simply run
```bash
python controller/train_controller.py --debug --cfg-override SAVE_DIR path/to/output/dir
```
