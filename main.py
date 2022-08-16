import enum
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from data_loader import SiameseNetworkDataset
from models import STCN
from customized_criterion import ContrastiveLoss
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

train_data_path = "dataset/snn_train.npy"
test_data_path = "dataset/snn_test.npy"
with open(train_data_path, 'rb') as f:
    train_data = np.load(f)
with open(test_data_path, 'rb') as f:
    test_data = np.load(f)

train_X = train_data[:, :train_data.shape[1]-1]
train_y = train_data[:, -1]

test_X = test_data[:, :test_data.shape[1]-1]
test_y = test_data[:, -1]

mean_train_X = train_X.mean(axis=0)
std_train_X = train_X.std(axis=0)

train_X = (train_X - mean_train_X) / std_train_X
test_X = (test_X - mean_train_X) / std_train_X

training_set = np.concatenate((train_X, train_y.reshape(-1,1)) ,axis=1)
test_set = np.concatenate((test_X, test_y.reshape(-1,1)) ,axis=1)

batch_size = 32
train_siamese_dataset = SiameseNetworkDataset(training_set)
my_train_dataloader = DataLoader(train_siamese_dataset, shuffle=True, batch_size=batch_size)

print('===============================')
print('start training......')

Lr = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STCN(num_inputs=1, num_channels=[64, 64, 64, 64, 64, 64], dropout=0.1)
model.to(device)
summary(model)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr = Lr)


model.train()
counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(epochs):
    # Iterate over batches
    for i, (s1, s2, _, _, label) in enumerate(my_train_dataloader, 0):

        # Send the signals to devce(cpu or cuda)

        s1, s2, label = s1.float().to(device), s2.float().to(device), label.float().to(device)

        optimizer.zero_grad()

        output1, output2 = model(s1.view(s1.shape[0], 1, -1), s2.view(s2.shape[0], 1, -1))

        loss_contrastive = criterion(output1, output2, label)

        loss_contrastive.backward()

        optimizer.step()

        # if i % 100 == 0:
        #     print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
        #     iteration_number += 10

        #     counter.append(iteration_number)
        #     loss_history.append(loss_contrastive.item())

plt.plot(counter, loss_history)
plt.show()
