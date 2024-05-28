
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



# ---------------------------------------------------------------------
import torch
import numpy as np
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_inputs = tokenizer(sequences, padding=True, truncation=True, max_length=50, return_tensors="pt")

from torch.utils.data import DataLoader, Dataset, TensorDataset

labels = [np.random.randint(0,2) for _ in range(len(sequences))]
labels = torch.tensor(labels)


class MyDataset(Dataset):
    
    def __init__(self, labels, data):
        self.labels = labels
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    

dataset = MyDataset(labels, model_inputs["input_ids"])
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

dataset2 = TensorDataset(model_inputs["input_ids"], labels)
dataloader2 = DataLoader(dataset2, batch_size=8, shuffle=True)



import torch.nn as nn
class ToxicClassifierModel(nn.Module):
    
    def __init__(self, vocab_size, num_class, embedding_dim=50, hidden_size=128, num_LSTM_layers=2, fc_layers = [128, 256], bidirectional=True  ):
        super(ToxicClassifierModel, self).__init__()
        self.num_LSTM_layers = num_LSTM_layers
        self.bidirectional = bidirectional
        self.hidden_size= hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,
                             num_layers=num_LSTM_layers, bidirectional=bidirectional)
        fc_input_size = 2 * hidden_size if bidirectional==True else hidden_size
        self.fc1 = nn.Linear(fc_input_size, fc_layers[0])
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc3 = nn.Linear(fc_layers[1], num_class)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.embedding(x)

        h0 = torch.zeros((2 if self.bidirectional==True else 1) * self.num_LSTM_layers, 
                         x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros((2 if self.bidirectional==True else 1) * self.num_LSTM_layers, 
                         x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))


        # Take the output from the last time step (batch_size, seq_length, num_directions * hidden_size)
        out = out[:, -1, :] # This works if batch_first=True is used 

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        
        return out


vocab_size = tokenizer.vocab_size
num_class = 2
embedding_dim = 50
hidden_size=128
num_LSTM_layers=2
fc_layers = [128, 256]
bidirectional=True 

model = ToxicClassifierModel(vocab_size, num_class)
print(model)



import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss and Optimizer
optimizer = optim.Adam(params=)
