import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

print(activations[0].shape)
print(activations)