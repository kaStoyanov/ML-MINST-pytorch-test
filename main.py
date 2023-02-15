import torch,torchvision
from torch import nn
from torch import optim
from torch.optim import SGD
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Download dataset from here  www.di.ens.fr/~lelarge/MNIST.tar.gz
# Extract and move in the repo

# Class to load our datasets 
class CTDataset(Dataset):
    def __init__(self,filepath):
        self.x,self.y = torch.load(filepath)
        self.x = self.x/255 # Normalizing x data to 0-1
        self.y = F.one_hot(self.y,num_classes =10).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    
train_ds = CTDataset('./MNIST/processed/training.pt')
test_ds = CTDataset('./MNIST/processed/test.pt')
print(len(train_ds))
print(len(test_ds))

train_dl = DataLoader(train_ds, batch_size=5)
xs, ys = train_ds[0:4]


L = nn.CrossEntropyLoss()

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()
    
f = MyNeuralNet()
# print(L(f(xs), ys))

def train_model(dl, f, n_epochs=20):
    # Optimization
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update weights
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)

epoch_data, loss_data = train_model(train_dl,f)
print(loss_data)