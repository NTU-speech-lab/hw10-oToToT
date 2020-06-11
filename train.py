#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys

TRAIN_MODEL = True
USE_KNN     = True
V_WIDTH     = 200
BATCH_SIZE  = 128


# In[2]:


import torch
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda')


# In[3]:


import numpy as np
import torch 
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(880301)


# In[4]:


import os

def path_join(arr):
    r = arr[0]
    for i in range(1, len(arr)):
        r = os.path.join(r, arr[i])
    return r


# In[5]:


from torch import nn
from torch.autograd import Variable

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2



if TRAIN_MODEL:
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset, ConcatDataset
    import numpy as np
    import time

    num_epochs = 55
    learning_rate = 1e-5

    train = np.load(sys.argv[1], allow_pickle=True)
    x = train
        
    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=BATCH_SIZE,
                                  num_workers=(8 if os.name=='posix' else 0))


    model = conv_autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )
    
    best_loss = np.inf
    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for data in train_dataloader:
            img = data[0].transpose(3, 1).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================save====================
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model, sys.argv[2])
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f} {:2.2f} sec(s)'
              .format(epoch + 1, num_epochs, loss.item(), time.time() - epoch_start_time))

