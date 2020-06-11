#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys


USE_KNN     = ('best.pth' in sys.argv[2])
USE_NAIVE   = ('baseline.pth' in sys.argv[2])
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



from torch import nn
from torch.autograd import Variable

if USE_KNN:
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
else:
	class conv_autoencoder(nn.Module):
		def __init__(self):
			super(conv_autoencoder, self).__init__()
			self.encoder = nn.Sequential(
				nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
				nn.ReLU(),
				nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
				nn.ReLU(),
				nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
				nn.ReLU(),
			)
			self.decoder = nn.Sequential(
				nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
				nn.ReLU(),
				nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
				nn.ReLU(),
				nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
				nn.Tanh(),
			)

		def forward(self, x):
			x = self.encoder(x)
			x = self.decoder(x)
			return x

if USE_NAIVE:
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
    import numpy as np
    
    test = np.load(sys.argv[1], allow_pickle=True)
    y = test
    
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=BATCH_SIZE, 
                                 num_workers=(16 if os.name=='posix' else 0))

    model = torch.load(sys.argv[2]).to(DEVICE)

    model.eval()
    reconstructed = []
    with torch.no_grad():
        for data in test_dataloader:
            output = model(data[0].transpose(3, 1).to(DEVICE))
            reconstructed.append(output.reshape(len(output), -1).cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    y_code = reconstructed
    y_cluster = np.zeros(len(y_code))
    
    anomality = np.sqrt(np.sum(np.square(reconstructed - y.reshape(len(y), -1)).reshape(len(y), -1), axis=1))
    y_pred = anomality
    with open(sys.argv[3], 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))


# In[11]:


if USE_KNN:
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
    from scipy.cluster.vq import vq, kmeans
    import numpy as np

    test = np.load(sys.argv[1], allow_pickle=True)
    y = test
    
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=BATCH_SIZE, 
                                 num_workers=(16 if os.name=='posix' else 0))
    
    
    model = torch.load(sys.argv[2]).to(DEVICE)

    model.eval()
    y_code, reconstructed = [], []
    for i, data in enumerate(test_dataloader): 
        img = data[0].transpose(3, 1).cuda()
        output = model(img).transpose(3, 1)
        code = model.encode(img)
        y_code.append(code.detach().cpu().numpy())
        reconstructed.append(output.detach().cpu().numpy())

    y_code = np.concatenate(y_code, axis=0)
    y_code = y_code.reshape(len(y_code), -1)
    
    print(hash(str(y_code)))
    
    reconstructed = np.concatenate(reconstructed, axis=0)
    reconstructed = reconstructed.reshape(len(reconstructed), -1)
    
    kmeans = MiniBatchKMeans(n_clusters=6, batch_size=100).fit(y_code)
    y_cluster = kmeans.predict(y_code)
    y_dist = np.sum(np.square(kmeans.cluster_centers_[y_cluster] - y_code), axis=1)
    y_pred = y_dist


    with open(sys.argv[3], 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

