nworkers=32
rdma=0
opt=acpsgd # choices: ssgd, powersgd, acpsgd

opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet50 bs=64 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=4 dnn=resnet152 bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=32 dnn=bert_base bs=32 bash perf.sh
#opt=$opt rdma=$rdma nworkers=$nworkers rank=32 dnn=bert_large bs=8 bash perf.sh

