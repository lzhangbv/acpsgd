## Iteration time comparison
nworkers=32
rdma=0
opt=acpsgd # choices: ssgd, powersgd, acpsgd
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet152 bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=32 dnn=bert_large bs=8 bash perf.sh

## effect of buffer size
opt=acpsgd # choices: ssgd, powersgd, acpsgd
rank=32
#threshold=0 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh
#threshold=25 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh
#threshold=50 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh
#threshold=100 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh
#threshold=500 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh
#threshold=1000 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh
#threshold=1500 opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=bert_large bs=8 bash perf.sh

## effect of batch size
opt=acpsgd # choices: ssgd, powersgd, acpsgd
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet152 bs=8 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet152 bs=16 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet152 bs=24 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet152 bs=32 bash perf.sh

## effect of rank
opt=acpsgd # choices: ssgd, powersgd, acpsgd
#opt=$opt rdma=$rdma nworkers=$nworkers rank=32 dnn=bert_large bs=8 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=64 dnn=bert_large bs=8 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=128 dnn=bert_large bs=8 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=256 dnn=bert_large bs=8 bash perf.sh

## effect of the number of GPUs
opt=acpsgd # choices: ssgd, powersgd, acpsgd
#opt=$opt rdma=$rdma nworkers=4 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=$opt rdma=$rdma nworkers=8 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=$opt rdma=$rdma nworkers=16 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=$opt rdma=$rdma nworkers=32 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=$opt rdma=$rdma nworkers=64 rank=4 dnn=resnet50 bs=64 bash perf.sh

#opt=$opt rdma=$rdma nworkers=4 rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=8 rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=16 rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=32 rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=64 rank=32 dnn=bert_base bs=32 bash perf.sh

## effect of network bandwidth
rdma=0 # choices: 0 (10GbE), 1 (100GbIB), 2 (1GbE)
#opt=ssgd rdma=$rdma nworkers=32 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=powersgd rdma=$rdma nworkers=32 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=acpsgd rdma=$rdma nworkers=32 rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=ssgd rdma=$rdma nworkers=32 rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=powersgd rdma=$rdma nworkers=32 rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=acpsgd rdma=$rdma nworkers=32 rank=32 dnn=bert_base bs=32 bash perf.sh

## benckmark communication time w. and w/o TF
rdma=0 # choices: 0 (10GbE), 1 (100GbIB), 2 (1GbE)
#node_count=8 ngpu_per_node=4 node_rank=1 rdma=$rdma script=benckmark/comm_time.py bash launch_torch.sh

