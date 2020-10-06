# for i in range(10):
#    requests.post('http://127.0.0.1:5000', data={'mean_loss': i})

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

import torch.nn as nn
import torch.nn.functional as F

from common import Loader, conv4d


def normalization(x):
    return x / torch.max(x)


train = Loader('C:/Users/Milena/Documents/Renova/mammoneuro/dataset', 'train', normalize=True, normalize_func=normalization)
#val = Loader('mammoneuro/dataset', 'val', normalize=True, normalize_func=normalization)
print('end of loading')


def report(number_of_step, N, lst_of_lst):
    result = f'{number_of_step}'
    for lst in lst_of_lst:
        if len(lst) < N:
            curr_loss_history = lst
        else:
            curr_loss_history = lst[-N:]

        s = f'min: {min(curr_loss_history):.4f}\t mean: {(sum(curr_loss_history) / len(curr_loss_history)):.4f}\t max: {max(curr_loss_history):.4f}'
        result = f'{result} \t___\t{s}'
    print(result)


class ModelSimplest(nn.Module):
    def __init__(self):
        super(ModelSimplest, self).__init__()
        self.name = 'torch_model_4'

        self.head = nn.Sequential(
            conv4d(1, 3, 13),
            nn.ReLU(inplace=True)
        )


        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(3 * (18 - 12) ** 4, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.out(self.head(x))


url = 'http://127.0.0.1:5000'


def random(x, mx=0.2):
    r = torch.normal(0.0, 1.0, size=x.size()).cuda()
    r = r / torch.max(r)
    return r * mx


cel = nn.BCELoss()


def compute_grad(model, x, target):
    y = model(x)
    loss = cel(y, target)
    loss.backward()
    return loss.item()


def train_step(model, optimizer, scheduler, reduce_scheduler, batch_size=64):
    model.train()
    number_of_step = 0

    losses = []
    for filename, x, target in train.generator(batch_size):
        optimizer.zero_grad()

        losses.append(compute_grad(model, x, target))

        optimizer.step()
        scheduler.step()

        if number_of_step % 5 == 0:
            if number_of_step != 0:
                data = requests.post(url, data={'mean_loss': sum(losses[-5:]) / 5,
                                                'min_loss': min(losses[-5:]), 'max_loss': max(losses[-5:])})
                # , 'lrates': [g['lr'] for g in optimizer.param_groups]})
            else:
                data = requests.post(url,
                                     data={'mean_loss': losses[-1], 'min_loss': losses[-1], 'max_loss': losses[-1],
                                           'lrates': [g['lr'] for g in optimizer.param_groups]})

        if number_of_step % 30 == 0:
            report(number_of_step, 30 * 4, [losses])
            reduce_scheduler.step(sum(losses[-4 * 30:]))

        number_of_step += 1

        if number_of_step == 180:
            break

    return optimizer


print('model initiation')
model = ModelSimplest()

print('other stuff')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6 * 2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.975, last_epoch=-1)
reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

print('learning')
batch_size = 128
for _ in range(10):
    optimizer = train_step(model, optimizer, scheduler, reduce_scheduler, batch_size=batch_size)