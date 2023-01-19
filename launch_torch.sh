#!/bin/bash

# usage:
# ngpu_per_node=4 node_count=2 rdma=1 script=$script params=$params bash launch_torch.sh

# python env and script
source envs.conf

torchrun="torch.distributed.run"
#torchrun="torch.distributed.launch" #deprecated in pytorch1.10.0

directory=`pwd`
script="${script:-}"
params="${params:-}"
echo "launch dir: $directory"

# network config
rdma="${rdma:-0}"
if [ "$rdma" = "0" ]; then
    net_config="export NCCL_SOCKET_IFNAME=$ETH_INTERFACE; export NCCL_IB_DISABLE=1;"
elif [ "$rdma" = "1" ]; then
    net_config="export NCCL_SOCKET_IFNAME=$IB_INTERFACE; export NCCL_IB_DISABLE=0;"
else
    net_config="export NCCL_SOCKET_IFNAME=$ETH_INTERFACE_1Gb; export NCCL_IB_DISABLE=1;"
fi
net_config="export OMP_NUM_THREADS=4; $net_config"

#use ring algorithm
#net_config="export NCCL_ALGO=Ring; $net_config"

# cluster settings
total_host=$total_host
hosts=$hosts

# multi-node multi-gpu settings
ngpu_per_node="${ngpu_per_node:-1}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"

node_rank=$(expr $node_rank - 1) # array index
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi
master_host=${hosts[$node_rank]}
node_rank_copy_=$node_rank

i=0
while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    #args="$PY -m torch.distributed.launch --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host $script $params" #deprecated in pytorch1.10.0
    args="$PY -m $torchrun --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host $script $params"
    echo "$host: $args"
    cmd="cd $directory; $net_config $args"
    if [ $(expr $i + 1) -eq $node_count ]; then
        ssh $host $cmd   # return until finished or interrupted
    else
        ssh $host $cmd & # return immediately
    fi
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done

remote_kill=1
if [ $remote_kill -eq 1 ]; then
#echo "kill remote launched torch processes..."
node_rank=$node_rank_copy_
i=0
while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    cmd="kill -9 \$(ps aux|grep $script | awk '{print \$2}')" # with escaping \$
    #echo "$host: $cmd"
    ssh $host $cmd
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done
fi

