'''
Author: Qi7
Date: 2022-08-17 14:29:38
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2022-08-17 15:32:58
Description:
'''
#%%
import numpy as np
import torch
from matplotlib import pyplot as plt
from models import STCN
from data_loader import SiameseNetworkDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

# load the loss history
fig_path = "figs/"
history = np.load("train_loss_history.npy")


#%%
plt.figure(figsize=(20,20))
plt.plot(history[0:-1:7])
plt.title('loss curve', fontsize=26)
plt.xlabel("epoch number", fontsize=24)
plt.ylabel("Loss", fontsize=24)

plt.savefig(fig_path+'loss_curve.svg')
plt.close()
# %%

# loading test data set
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
test_set = np.concatenate((test_X, test_y.reshape(-1,1)) ,axis=1)


# loading the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model_path = ""
model = STCN(num_inputs=1, num_channels=[64, 64, 64, 64, 64, 64], dropout=0.1)
model.load_state_dict(torch.load("current_epoch_model.pth"))
model.to(device)

model.eval()
batch_size = 1
test_siamese_dataset = SiameseNetworkDataset(test_set)
my_test_dataloader = DataLoader(test_siamese_dataset, shuffle=True, batch_size=batch_size)

# Grab one signal that we are going to test
dataiter = iter(my_test_dataloader)
# x0, _, label0 = next(dataiter)

# randomly choose ten samples
for i in range(10):
    x0, x1, label0, label1, _ = next(dataiter)
    output1, output2 = model(x0.view(x0.shape[0], 1, -1).float().to(device), x1.view(x0.shape[0], 1, -1).float().to(device))
    euclidean_distance = F.pairwise_distance(output1, output2)
    plt.figure(figsize=(10,10))
    plt.plot(x0.detach().numpy().reshape(-1))
    plt.show()
    print(f"Unsimilarity between: {euclidean_distance.item():.2f}")
    print(f"x0 label: {label0.item()}, x1 label:{label1.item()}")
    plt.figure(figsize=(10,10))
    plt.plot(x1.detach().numpy().reshape(-1))
    plt.show()
# %%
