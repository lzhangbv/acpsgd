nworkers=8
rdma=0

dnn=resnet50
bs=64
rank=4
opt=ssgd # choices: ssgd, powersgd, acpsgd

opt=$opt rdma=$rdma nworkers=$nworkers rank=$rank dnn=$dnn bs=$bs bash perf.sh

