#!/bin/bash
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"

source envs.conf
script=examples/mnist/pytorch_mnist.py
params=''

# multi-node multi-GPU setting
node_rank=16  # launch node1, node2, ...
ngpu_per_node=4
node_count=$(expr $nworkers / $ngpu_per_node)

if [ $nworkers -lt 4 ]; then # single-node
    ngpu_per_node=$nworkers node_count=1 node_rank=$node_rank rdma=$rdma script=$script params=$params bash launch_torch.sh
else
    ngpu_per_node=$ngpu_per_node node_count=$node_count node_rank=$node_rank rdma=$rdma script=$script params=$params bash launch_torch.sh
fi

