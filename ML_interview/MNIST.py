import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.getcwd()
df = pd.read_csv("../../MNIST Data( Digit Recognizer)/data/raw/train.csv")

X = df.drop("label", axis=1)
y = df["label"]


plt.figure(figsize=(4,4))
plt.imshow(X.loc[1].to_numpy().reshape(28,28), cmap="Greys")
plt.title("An Example")
plt.show()



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




from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
