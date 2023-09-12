# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import time
import torch.nn as nn
import os
import numpy as np
import torch
import torch.optim as optim
import time
import torch.nn as nn
import logging
import os
import arrow
from pathlib import Path
import numpy as np


def SequenceMask(X, X_len,value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :].to(X_len.device) < X_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length).float()
        self.reduction='none'
        output=super(MaskedSoftmaxCELoss, self).forward(pred.transpose(1,2), label)
        return (output*weights).mean(dim=1)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer    # 优化器
        self._step = 0                # 步长
        self.warmup = warmup          # warmup_steps
        self.factor = factor          # 学习率因子(就是学习率前面的系数)
        self.model_size = model_size  # d_model
        self._rate = 0                # 学习率

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def grad_clipping(params, theta, device):
    """Clip the gradient."""
    norm = torch.tensor([0], dtype=torch.float32, device=device)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)

def grad_clipping_nn(model, theta, device):
    """Clip the gradient for a nn model."""
    grad_clipping(model.parameters(), theta, device)

# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def train(model, data_iter, lr, factor, warmup, num_epochs, device, model_path, logger,data_test):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = NoamOpt(model.enc_net.input_size, factor, warmup,
    #                    torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #loss = MaskedSoftmaxCELoss()
    loss = nn.MSELoss()
    tic = time.time()
    pre = []
    true = []
    min = 0
    for epoch in range(1, num_epochs + 1):
        l_sum, num_tokens_sum = 0.0, 0.0
        lval_sum = 0
        model.train()
        for step, sample in enumerate(data_iter):
            optimizer.zero_grad()
            X = sample['X'] #[8,4,6]
            Y = sample['Y']
            # X = X.cuda()
            # Y = Y.cuda()
            X = X.to(device)
            Y = Y.to(device)
            # obs_half = obs_half.to(device)

            # Y_hat, _ = model(X, X)
            Y_hat,_ = model(X[:,:,0:1],X,X[:,:,2:3])
            l = loss(Y_hat.squeeze(1), Y.squeeze(1))
            l.backward()
            #with torch.no_grad():
             #   grad_clipping_nn(model, 5, device)
            optimizer.step()
            l_sum += l.sum().item()
            print(f'epoch:{epoch}step{step}loss{l}')
            if epoch == num_epochs:
                pre.append(Y_hat.squeeze(1))
                true.append(Y.squeeze(1))
        print(f"Epoch {epoch :03d} | Train loss: {l_sum/(step+1)}")

        model.eval()
        with torch.no_grad():
            loss = nn.MSELoss()
            pre = []
            true = []
            l_sum, num_tokens_sum = 0.0, 0.0
            for step, sample in enumerate(data_test):
                X = sample['X']  # [1,4,1]
                Y = sample['Y']  # [1,1]
                # X = X.cuda()
                # Y = Y.cuda()
                X = X.to(device)
                Y = Y.to(device)
                # obs_half = obs_half.to(device)
                # Y_hat, _ = model(X, X)
                Y_hat,_ = model(X[:,:,0:1],X,X[:,:,2:3])
                l = loss(Y_hat.squeeze(1), Y.squeeze(1))
                # num_tokens = Y_vlen.sum().item()
                l_sum += l.sum().item()
                # logger.info(f'step{step}loss{l_sum/(step+1)}')
                # num_tokens_sum += num_tokens
                pre.append(Y_hat.squeeze(1))
                true.append(Y.squeeze(1))
            correctness = (1 - np.sqrt(l_sum / (step + 1)))
            if min < correctness:
                min = correctness
                glb = list(model_path.glob('*.mdl'))
                for p in sorted(glb):
                    os.unlink(p)
                torch.save(model.state_dict(),
                           model_path / f'model_m_{correctness:0.6f}.epoch{epoch :03d}.mdl')
                pred = torch.cat(pre, dim=0)
                ober = torch.cat(true, dim=0)
            logger.info(f"Epoch {epoch :03d} | Train loss: {l_sum/(step+1)} | correctness:{correctness}")
    return pred, ober

def test(model, data_iter, lr, factor, warmup, num_epochs, device, model_path, logger):
    """Translate based on an encoder-decoder model with greedy search."""
    glb = list(model_path.glob('*.mdl'))
    if glb!=[]:
        checkpoint = torch.load(sorted(glb)[-1])
        model.load_state_dict(checkpoint)
    # checkpoint = torch.load('model/_-20211206_210240/model_m_0.861180.epoch040.mdl')
    #model.load_state_dict(checkpoint)
    model.to(device)
    with torch.no_grad():
        loss = nn.MSELoss()
        pre = []
        true = []
        l_sum, num_tokens_sum = 0.0, 0.0
        for step, sample in enumerate(data_iter):
            X = sample['X']  # [8,4,6]
            Y = sample['Y']
            # X = X.cuda()
            # Y = Y.cuda()
            X = X.to(device)
            Y = Y.to(device)
            # obs_half = obs_half.to(device)
            # Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen - 1

            # Y_hat, _ = model(X, X)
            Y_hat,_ = model(X[:,:,0:1],X,X[:,:,2:3])
            l = loss(Y_hat.squeeze(1), Y.squeeze(1))
            # num_tokens = Y_vlen.sum().item()
            l_sum += l.sum().item()
            logger.info(f'step{step}loss{l_sum/(step+1)}')
            # num_tokens_sum += num_tokens
            pre.append(Y_hat.squeeze(1))
            true.append(Y.squeeze(1))
        #logger.info(f"Epoch {epoch + 1:03d} | Test loss: {l_sum/(step+1)}")

    return torch.cat(pre, dim=0), torch.cat(true, dim=0)
