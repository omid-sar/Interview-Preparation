
# ---------------------------------------------- Downlaod ----------------------------------------------
import requests

url = "https://raw.githubusercontent.com/omid-sar/Project_Data_Hub/main/twitter_toxic_final_balanced_dataset.csv"
response = requests.get(url)

if response.status_code == 200:
    with open("../data/data_manipulation_resources/twitter_toxic.csv", "wb") as file:
        file.write(response.content)
    print("File downloaded and saved successfully")
else:
    print(f"Failed download successfully. Status code: {response.status_code}")

# ----------------------------------------------- Read --------------------------------------------------
import pandas as pd
import numpy as np

df = pd.read_csv("../data/data_manipulation_resources/twitter_toxic.csv")
df = df.drop(columns=["Unnamed: 0"])
df.head(2)

# ----------------------------------------------- EDA -----------------------------------------------------
import matplotlib.pyplot as plt
from  transformers import BertTokenizer

df["tweet_length"] = [len(seq.split()) for seq in df["tweet"]]
toxic_len = df[df["Toxicity"] == 1]["tweet_length"]
neutral_len = df[df["Toxicity"] == 0]["tweet_length"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
df["token_lenght"] = [len(tokenizer(seq).input_ids) for seq in df["tweet"]]
toxic_token_len = df[df["Toxicity"] == 1]["token_lenght"]
neutral_token_len = df[df["Toxicity"] == 0]["token_lenght"]

fig, axes = plt.subplots(2,1 , figsize=(14,6))
axes[0].hist(toxic_len, bins=100, alpha=0.5, label='The Length of Toxic Tweets')
axes[0].hist(neutral_len, bins=100, alpha=0.5, label='The Length of Neutral Tweets')
axes[0].legend()

axes[1].hist(toxic_token_len, bins=100, alpha=0.5, label='The Length of Toxic Tweet Tokens')
axes[1].hist(neutral_token_len, bins=100, alpha=0.5, label='The Length if Neutral Tweet Tokens')
axes[1].set_xlim([0, 120])
axes[1].legend()
plt.show()

# ----------------------------------------------- Dataset -----------------------------------------------
import torch
from torch.utils.data import  Dataset, TensorDataset

class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

X = [seq for seq in df["tweet"]]
y = [torch.tensor(seq) for seq in df["Toxicity"]]

MAX_LENGTH=64
model_input = tokenizer(X, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
dataset = MyDataset(data=model_input.input_ids, labels=y)

# ----------------------------------------------- Data Loader ------------------------------------------------
from torch.utils.data import random_split, DataLoader

train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, val_size, test_size])

batch_size = 32
num_workers = 2
dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
next(iter(dataloader))

# ----------------------------------------------- Downlaod ------------------------------------------------
import torch.nn as nn

class LSTM(nn.Module):
    def __init(self, vocab_size, embedding_dim=64, bidirectional=True, lstm_hidden_size=128,
                num_lstm_layers=2, hidden_layers=[256,128], num_classes=2):
        super(LSTM, self).__init__()
        self.bidirectional = bidirectional
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, num_layers =num_lstm_layers, 
                            batch_fist=True, bidirectional=bidirectional)
        fc_input_size = 2 * lstm_hidden_size if bidirectional else lstm_hidden_size
        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embeddings(x)

        h0 = torch.zeros((2 if self.bidirectional else 1) * self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(device)
        h0 = torch.zeros((2 if self.bidirectional else 1) * self.num_lstm_layers, x.size(0), self.lstm_hidden_size).to(device)

        out = self.lstm()




# ----------------------------------------------- Downlaod ------------------------------------------------