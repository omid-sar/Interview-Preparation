import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


my_data = pd.read_csv('../data/data_manipulation_resources/twitter_toxic_final_balanced_dataset.csv', usecols=["Toxicity", "tweet"])
my_data.loc[1]

from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
dataset = MyDataset(my_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
first_batch = next(iter(dataloader))
print(first_batch)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*13*13, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32*13*13)
        x = self.fc1(x)
        return x


model = SimpleCNN().to(device)

activations = []

def hook_fn(module, input, output):
    activations.append(output)

def hook_fn(module, input, output):
    print(f"Inside {module.__class__.__name__} forward hook")
    input_shapes = [inp.shape for inp in input]
    print(f"Input shapes: {input_shapes}")
    print(f"Output shape: {output.shape}")


# Register the forward hook to the conv1 layer 
model.conv1.register_forward_hook(hook_fn)

input_data = torch.randn(1, 1, 28 , 28)

output = model(input_data)

#print(activations[0].shape)
print(activations)



class LSTM(nn.Module):
    def __init__(self, )
        
import numpy as np
np.random.seed(42)
m, n = 16, 100
X = np.random.randn(m,n)
X = np.random.rand(2,4)


import numpy as np
np.random.seed(42)
m, n = 16, 100
print(np.random.randn(m,n))
print(np.random.randn(m,n))