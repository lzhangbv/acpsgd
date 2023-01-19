import torch
import torchvision.models as models
from transformers import BertForPreTraining, BertConfig
import time
import numpy as np
import torch.distributed as dist
import os

# init
dist.init_process_group(backend='nccl', init_method='env://')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# model
model_name = 'bert_base'

rank = 32 if (model_name == 'bert_base' or model_name == 'bert_large') else 4
if model_name == 'resnet50':
    model = models.resnet50()
elif model_name == 'resnet152':
    model = models.resnet152()
else:
    dirname = os.path.dirname(__file__)
    bert_filename = os.path.join(dirname, 'bert_base_config.json') if model_name == 'bert_base' else os.path.join(dirname, 'bert_large_config.json')
    config = BertConfig.from_json_file(bert_filename)
    model = BertForPreTraining(config)

model.cuda()

# tensors
Ms = []
Ps = []
Qs = []
M_size = 0
P_size = 0
Q_size = 0

for p in model.parameters():
    if p.requires_grad: 
        if p.ndimension() > 1: 
            M = p.view(p.shape[0], -1)
            r = min(rank, min(M.shape))
            Ps.append(torch.randn(M.shape[0]*r, device=p.device))
            Qs.append(torch.randn(M.shape[1]*r, device=p.device))
            Ms.append(M)
            P_size += M.shape[0]*r
            Q_size += M.shape[1]*r
            M_size += M.numel()
        else:
            Ps.append(p)
            Qs.append(p)
            Ms.append(p)
            P_size += p.numel()
            Q_size += p.numel()
            M_size += p.numel()

M_buffer = [torch.randn(M_size, device=p.device)]
P_buffer = [torch.randn(P_size, device=p.device)]
Q_buffer = [torch.randn(Q_size, device=p.device)]

# benckmark communication
def benckmark_step(tensors, step=5):
    iter_times = []
    for i in range(step+1):
        stime = time.time()
        for tensor in tensors:
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        iter_times.append(time.time() - stime)
    return np.mean(iter_times[1:]) * 1000

# Ms vs. M_buffer
Ms_iter_time = benckmark_step(Ms)
Mb_iter_time = benckmark_step(M_buffer)
if dist.get_rank() == 0:
    print("# of Ms: %d, # of elements: %d" % (len(Ms), M_size))
    print("Time (ms) of Ms: %.3f, M buffer: %.3f" % (Ms_iter_time, Mb_iter_time))

# Ps vs. P_buffer
Ps_iter_time = benckmark_step(Ps)
Pb_iter_time = benckmark_step(P_buffer)
if dist.get_rank() == 0:
    print("# of Ps: %d, # of elements: %d" % (len(Ps), P_size))
    print("Time (ms) of Ps: %.3f, P buffer: %.3f" % (Ps_iter_time, Pb_iter_time))

# Qs vs. Q_buffer
Qs_iter_time = benckmark_step(Qs)
Qb_iter_time = benckmark_step(Q_buffer)
if dist.get_rank() == 0:
    print("# of Qs: %d, # of elements: %d" % (len(Qs), Q_size))
    print("Time (ms) of Qs: %.3f, Q buffer: %.3f" % (Qs_iter_time, Qb_iter_time))

