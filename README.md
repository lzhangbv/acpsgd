# Evaluation and Optimization of Gradient Compression for Distributed Deep Learning

## Introduction 
We propose a new gradient compression algorithm called ACP-SGD, that alternates the low-rank compression and aggregation between PowerSGD's P and Q, so as to enable system optimization techniques such as ring all-reduce, pipelining and tensor fusion. This repository contains ACP-SGD's source code (see [acpsgd.py](https://github.com/lzhangbv/powersgd/blob/main/acpsgd.py)), as well as a set of benchmarking scripts for evaluating the training performance among S-SGD, Power-SGD, and ACP-SGD. 

Currently, it covers: 
### Data Parallelism Algorithms
- S-SGD atop [PyTorch-DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Power-SGD atop [communication hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html)
- ACP-SGD, which supports tensor fusion with hyper-parameters rank and threshold (default: 25MB) 

We refer readers to [gradient_reducers](https://github.com/epfml/powersgd/blob/master/paper-code/gradient_reducers.py) for evaluating more gradient compression methods, such as Top-k SGD and Sign-SGD. 

### Deep Neural Networks
- [Convolutional neural networks (CNNs)](https://pytorch.org/vision/stable/models.html) on a fake ImageNet data set (i.e., randomly generate the input image of 224\*224\*3)
- [Transformers](https://github.com/huggingface/transformers): BERT-Base and BERT-Large pretraining models.

## Installation
### Prerequisites
- Python 3.6+
- CUDA-10.+
- NCCL-2.4.+
- [PyTorch-1.12.+](https://download.pytorch.org/whl/torch_stable.html)

### Configure the cluster settings
Before running the scripts, please carefully configure the configuration file envs.conf, e.g.
- PY: python environment
- xxx_INTERFACE: network
- hosts: cluster configuration

## Run benchmarks
- The batch mode
```
bash batch.sh
```

- The individual mode, e.g.,
```
opt=acpsgd rank=4 dnn=resnet50 bs=64 nworkers=32 bash perf.sh
```

For different experimental settings, users can modify the algorithm, DNN model, batch size, the number of GPUs, and network configurations. 


## ACP-SGD Usage
The ACP-SGD distributed optimizer can be easily used like `horovod.DistributedOptimizer()`.
```Python
import acpsgd
acpsgd.init()
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = acpsgd.DistributedOptimizer(optimizer, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
...
```

### ACP-SGD Example
Example script for training on MNIST was provided.
```
$ bash mnist.sh
```

## Paper
If you are using this repository for your paper, please cite our work
```
@article{lin23acpsgd,
    author = {Zhang, Lin and Zhang, Longteng and Shi, Shaohuai and Chu, Xiaowen and Li, Bo},
    title = {Evaluation and Optimization of Gradient Compression for Distributed Deep Learning},
    journal = {IEEE International Conference on Distributed Computing Systems (ICDCS)},
    year = {2023}
}
```
