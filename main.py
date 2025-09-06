import numpy as np
import torch
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from data_sampler import DataSampler
from torch.utils.data import DataLoader
from model import PlaceNet
from arguments import get_args
import torch.optim as optim


args = get_args()

# init dataloader
data_sampler_train = DataSampler(split='train')
train_loader = DataLoader(data_sampler_train, batch_size=None)

data_sampler_val = DataSampler(split='val')
val_loader = DataLoader(data_sampler_val, batch_size=None)

# init model
model = PlaceNet().to(args.device)

# init loss
ce_loss = nn.CrossEntropyLoss()

# init adam
optimizer = optim.Adam(model.parameters(), lr=args.LR, betas=(args.Beta1, args.Beta2), eps=args.eps, weight_decay=0)

def xnor(a, b):
    if not (a ^ b):
        return 1
    else:
        return 0

epoch_loss_list = []
acc_list = []
# start training
for e in range(args.epoches):
    epoch_loss = 0
    iter_num = 0
    for datas in train_loader:
        model.train()
        data = datas[0].float().to(args.device)
        label = datas[1].to(args.device)

        output = model(data)

        loss = ce_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss {}'.format(loss.detach().item()))
        epoch_loss += loss.detach().item()
        iter_num += 1

    print('epoch {} loss: {}'.format(e, epoch_loss / iter_num))
    epoch_loss_list.append(epoch_loss / iter_num)
    correct_num = 0
    total_num = 0
    for datas in val_loader:
        model.eval()
        data = datas[0].float().to(args.device)
        label = datas[1].to(args.device).item()
        output = model(data)
        pred_y = torch.max(output, dim=1)[1].item()
        correct_num += xnor(pred_y, label)
        total_num += 1
    print('ACC {}'.format(correct_num/total_num))
    acc_list.append(correct_num/total_num)
    # break
print(epoch_loss_list)
print(acc_list)
plt.subplot(1,2,1)
plt.plot(epoch_loss_list)
plt.subplot(1,2,2)
plt.plot(acc_list)
plt.show()