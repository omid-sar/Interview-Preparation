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



from torch.utils.data import DataLoader, Dataset, TensorDataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index].to_numpy().astype(float)
        y = self.labels.iloc[index]
        return x,y
    



dataset = MyDataset(X,y )
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





























sequences = [' @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run',
 "@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked",
 '  bihday your majesty',
 '#model   i love u take with u all the time in urð\x9f\x93±!!! ð\x9f\x98\x99ð\x9f\x98\x8eð\x9f\x91\x84ð\x9f\x91\x85ð\x9f\x92¦ð\x9f\x92¦ð\x9f\x92¦  ',
 ' factsguide: society now    #motivation',
 '[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo  ',
 ' @user camping tomorrow @user @user @user @user @user @user @user dannyâ\x80¦',
 "the next school year is the year for exams.ð\x9f\x98¯ can't think about that ð\x9f\x98\xad #school #exams   #hate #imagine #actorslife #revolutionschool #girl",
 'we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â\x80¦ ',
 " @user @user welcome here !  i'm   it's so #gr8 ! ",
 ' â\x86\x9d #ireland consumer price index (mom) climbed from previous 0.2% to 0.5% in may   #blog #silver #gold #forex',
 'we are so selfish. #orlando #standwithorlando #pulseshooting #orlandoshooting #biggerproblems #selfish #heabreaking   #values #love #',
 'i get to see my daddy today!!   #80days #gettingfed',
 "@user #cnn calls #michigan middle school 'build the wall' chant '' #tcot  ",
 'no comment!  in #australia   #opkillingbay #seashepherd #helpcovedolphins #thecove  #helpcovedolphins',
 'ouch...junior is angryð\x9f\x98\x90#got7 #junior #yugyoem   #omg ','i am thankful for having a paner. #thankful #positive' ]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model_inputs = tokenizer(sequences, padding=True, truncation=True, max_length=32, return_tensors='pt')

print("tweet: \n " , sequences[0],
      "\n input_ids: \n",  model_inputs.input_ids[0],
      "\n token_type_ids: \n" , model_inputs.token_type_ids[0],
      "\n attention_mask: \n" , model_inputs.attention_mask[0])

model_inputs.keys()

import numpy as np
y = [np.random.randint(0,2) for _ in range(len(sequences))]
labels = torch.Tensor(y)

from torch.utils.data import  TensorDataset, DataLoader, random_split
dataset = TensorDataset(model_inputs["input_ids"], model_inputs["attention_mask"], labels)
dataset1 = TensorDataset(model_inputs.input_ids, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size -val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataset