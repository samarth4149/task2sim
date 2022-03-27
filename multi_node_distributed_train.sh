#!/bin/bash
# NODE_RANK=<populate>
# MASTER=<populate with master IP>
NUM_NODES=$1
NUM_PROC=$2
RAND_PORT=$3
echo "master addr = $MASTER , master port = $RAND_PORT"
shift 3
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_PROC --master_addr="$MASTER" --master_port=$RAND_PORT --node_rank=$NODE_RANK classifier/run_classifier.py "$@"
wait