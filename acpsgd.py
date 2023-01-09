# Copyright 2022 HKUST. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Implement ACP-SGD atop Pytorch. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import collections
import numpy as np
import torch.distributed as dist


def rank():
    return dist.get_rank()

def size():
    return dist.get_world_size()

THRESHOLD = 25 # default buffer size threshold (MB)

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, rank=4, threshold=THRESHOLD):
        r"""Distributed ACP-SGD optimizer with WFBP and tensor fusion. 
        Args:
            params: optimizer parameters. 
            named_parameters: A mapping between parameter names and values. 
            rank: rank size for power iteration. 
            threshold: buffer size threshold (in MB) for tensor fusion.
        """
        super(self.__class__, self).__init__(params)
        self._rank = rank
        self._threshold = threshold
        self._num_steps = 0
        self._compute_p = True  # compute P or Q
        self._grad_accs = []

        # parameter names
        if named_parameters is not None:
            named_parameters = list(named_parameters)
            self._param_names = {v: k for k, v in sorted(named_parameters)}
        else:
            self._param_names = {v: 'param.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}

        self._register_hooks()
        self._generate_groups_with_threshold()
        self._streams = {}
        if torch.cuda.is_available() and size() > 1:
            # Stream for grad reduction in the backward pass.  
            self._streams["reduce"] = torch.cuda.Stream()


    def _generate_groups_with_threshold(self):
        """
        Generate groups with buffer size threshold (in MB) for tensor fusion. 
        """
        p_size = []
        q_size = []
        model_size = []
        self._param_errors = {} # error-feedback for compressed gradients
        for p in self._register_parameters[::-1]:
            p_name = self._param_names[p]
            if p.ndimension() > 1: # compressed tensors
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape)) # ensure r<=min(n, m)
                self._param_errors[p_name] = torch.zeros_like(M)
                p_size.append(M.shape[0] * r * 4/1024/1024) #MB
                q_size.append(M.shape[1] * r * 4/1024/1024)
                model_size.append(p.data.numel() * 4/1024/1024)
            else: # uncompressed tensors
                p_size.append(p.data.numel() * 4/1024/1024)
                q_size.append(p.data.numel() * 4/1024/1024)
                model_size.append(p.data.numel() * 4/1024/1024)
        if rank() == 0: 
            print('Memory (MB) Grad: %.2f, P: %.2f, Q: %.2f'%(sum(model_size), sum(p_size), sum(q_size))) 
        
        # compressed buffer size
        threshold_p = self._threshold * sum(p_size) / sum(model_size)
        threshold_q = self._threshold * sum(q_size) / sum(model_size)

        # Generate P Groups e.g. [[p3, p2], [p1]]
        p_groups = []
        current_group = []
        tot_size = 0
        for i, p in enumerate(self._register_parameters[::-1]):
            ps = p_size[i]
            if tot_size == 0 or tot_size + ps <= threshold_p:
                current_group.append(p)
                tot_size += ps
            else:
                p_groups.append(current_group)
                current_group = [p]
                tot_size = ps
        if len(current_group) > 0:
            p_groups.append(current_group)
        self._prepare_tensor_fusion_p(p_groups)

        # Generate Q Groups e.g. [[p3, p2], [p1]]
        q_groups = []
        current_group = []
        tot_size = 0
        for i, p in enumerate(self._register_parameters[::-1]):
            qs = q_size[i]
            if tot_size == 0 or tot_size + qs <= threshold_q:
                current_group.append(p)
                tot_size += qs
            else:
                q_groups.append(current_group)
                current_group = [p]
                tot_size = qs
        if len(current_group) > 0:
            q_groups.append(current_group)
        self._prepare_tensor_fusion_q(q_groups)      


    def _prepare_tensor_fusion_p(self, p_groups):
        """
        Prepare tensor fusion based on groups. 
        """
        self._buffers_p = []            # P buffers
        self._param_ps = {}             # get P by name
        self._param_group_idx_p = {}    # get (group id, sub id) by name
        
        for group_idx, p_group in enumerate(p_groups):
            for sub_idx, p in enumerate(p_group):
                p_name = self._param_names[p]
                self._param_group_idx_p[p_name] = (group_idx, sub_idx)
            buffer_p = self._get_buffer_p(p_group)
            self._buffers_p.append(buffer_p)

        self._param_group_flags_p = [[False]*len(g) for g in p_groups] # check whether param group is ready

        if rank() == 0: 
            print('P Buffer sizes (MB):', 
                    ', '.join('{:.2f}'.format(buf.numel()*4/1024/1024) for buf in self._buffers_p))

    def _get_buffer_p(self, p_group):
        # buffer initialization
        start_p = 0
        for p in p_group:
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                start_p += M.shape[0] * r
            else:
                start_p += p.data.numel()
        buffer_p = torch.randn(start_p, device=p.device) # check: set seed

        # param_ps mapping
        start_p = 0
        for p in p_group:
            p_name = self._param_names[p]
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                ps = M.shape[0] * r
                P = buffer_p[start_p:start_p+ps].view(M.shape[0], r)
                self._param_ps[p_name] = P
                start_p += ps
            else:
                ps = p.data.numel()
                P = buffer_p[start_p:start_p+ps].view(p.data.shape)
                self._param_ps[p_name] = P
                start_p += ps
        return buffer_p

    def _prepare_tensor_fusion_q(self, q_groups):
        """
        Prepare tensor fusion based on groups. 
        """
        self._buffers_q = []            # Q buffers
        self._param_qs = {}             # get Q by name
        self._param_group_idx_q = {}    # get (group id, sub id) by name
        
        for group_idx, q_group in enumerate(q_groups):
            for sub_idx, p in enumerate(q_group):
                p_name = self._param_names[p]
                self._param_group_idx_q[p_name] = (group_idx, sub_idx)
            buffer_q = self._get_buffer_q(q_group)
            self._buffers_q.append(buffer_q)

        self._param_group_flags_q = [[False]*len(g) for g in q_groups] # check whether param group is ready

        if rank() == 0: 
            print('Q Buffer sizes (MB):', 
                    ', '.join('{:.2f}'.format(buf.numel()*4/1024/1024) for buf in self._buffers_q))

    def _get_buffer_q(self, q_group):
        # buffer initialization
        start_p = 0
        for p in q_group:
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                start_p += M.shape[1] * r
            else:
                start_p += p.data.numel()
        buffer_q = torch.randn(start_p, device=p.device) # check: set seed

        # param_ps mapping
        start_p = 0
        for p in q_group:
            p_name = self._param_names[p]
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                qs = M.shape[1] * r
                Q = buffer_q[start_p:start_p+qs].view(M.shape[1], r)
                self._param_qs[p_name] = Q
                start_p += qs
            else:
                qs = p.data.numel()
                Q = buffer_q[start_p:start_p+qs].view(p.data.shape)
                self._param_qs[p_name] = Q
                start_p += qs
        return buffer_q

    def _register_hooks(self):
        """
        Register hooks. 
        """
        self._register_parameters = []
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)
                    self._register_parameters.append(p)

    def _make_hook(self, p):
        """
        Add hooks for backward propagation. 
        """
        def hook(*ignore):
            assert not p.grad.requires_grad
            name = self._param_names.get(p)
            # get grad, and previous P, Q, Error-feedback
            tensor = p.grad.data
            P = self._param_ps.get(name)
            Q = self._param_qs.get(name)
            E = self._param_errors.get(name)

            if E is not None: # update compressed tensor in the buffer
                self._alternative_compress(name, tensor, P, Q, E)
            else: # update uncompressed tensor in the buffer
                if self._compute_p:
                    P.copy_(tensor)
                else:
                    Q.copy_(tensor)
            
            # check whether buffer is ready to call an all-reduce
            new_name, buffer = self._buffer_is_ready(name)

            if buffer is not None and size() > 1: 
                self._streams["reduce"].wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._streams["reduce"]):
                    dist.all_reduce(buffer)
        return hook

    def _alternative_compress(self, name, tensor, P, Q, E):
        with torch.no_grad():
            M = tensor.view(tensor.shape[0], -1)
            if self._compute_p: 
                Q.copy_(torch.linalg.qr(Q).Q)
                P.copy_((M + E) @ Q)
                E.add_(M - P @ Q.T)
            else:
                P.copy_(torch.linalg.qr(P).Q)
                Q.copy_((M + E).T @ P)
                E.add_(M - P @ Q.T)

    def _buffer_is_ready(self, name):
        """
        Check whether buffer is ready to call an all-reduce.
        """
        with torch.no_grad():
            if self._compute_p:
                group_idx, sub_idx = self._param_group_idx_p[name]
                self._param_group_flags_p[group_idx][sub_idx] = True
                for flag in self._param_group_flags_p[group_idx]:
                    if not flag: # not ready
                        return name, None
                buffer = self._buffers_p[group_idx]
                comm_name = 'reduce-p-group-%d' % group_idx
                return comm_name, buffer
            else:
                group_idx, sub_idx = self._param_group_idx_q[name]
                self._param_group_flags_q[group_idx][sub_idx] = True
                for flag in self._param_group_flags_q[group_idx]:
                    if not flag: # not ready
                        return name, None
                buffer = self._buffers_q[group_idx]
                comm_name = 'reduce-q-group-%d' % group_idx
                return comm_name, buffer

    def _bp_barrier(self):
        """
        Synchronize the all-reduce operations.
        """
        if size() > 1:
            torch.cuda.current_stream().wait_stream(self._streams["reduce"])

        # approximate gradient
        for p in self._register_parameters:
            if p.ndimension() > 1: 
                name = self._param_names.get(p)
                # get P and Q
                P = self._param_ps.get(name)
                Q = self._param_qs.get(name)
                p.grad.data.view(p.grad.shape[0], -1).copy_(P @ Q.T)
            else:
                name = self._param_names.get(p)
                bias = self._param_ps.get(name) if self._compute_p else self._param_qs.get(name)
                p.grad.data.copy_(bias)
            p.grad.data.div_(size())
        
        # clear flags
        self._param_group_flags_p = [[False]*len(g) for g in self._param_group_flags_p]
        self._param_group_flags_q = [[False]*len(g) for g in self._param_group_flags_q]
        self._compute_p = not self._compute_p


    def step(self, closure=None): 
        """
        Performs a single optimization step.
        """
        self._bp_barrier()
        self._num_steps += 1
        return super(self.__class__, self).step(closure)



def DistributedOptimizer(optimizer, named_parameters=None, rank=4, threshold=THRESHOLD):
    """
    Wrap optimizer to gurantee the consistency. 
    Warning: some functions are not supported now, so we will simply skip these parameters.
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, rank, threshold)

def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    for name, p in params:
        if p is not None:
            dist.broadcast(p.view(-1), root_rank)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if p is not None and not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, _ in params:
        if key in callbacks:
            callbacks[key]()

def allreduce(tensor, name=None):
    dist.all_reduce(tensor)
    return tensor/dist.get_world_size()
