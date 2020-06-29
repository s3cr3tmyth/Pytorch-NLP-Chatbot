import json
from utils import tokenize, stem, bow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

# print(intents) 
all_words = []
tags = []
xy = []

for intent in intents['intent']:
    tag = intents['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        # use extend instead of append as we don;t want array of arrays
        xy.append((w,tag))

ignore_words = ['?','!','[',']','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (sen, tag) in xy:
    bag = bow(sen,all_words)
    X_train.append(bag)

    # multiclass label
    labels = tags.index(tag)
    y_train.append(labels)

y_train = np.array(y_train)
X_train = np.array(X_train)


class chatDatset(Dataset):
    def __init__(self):
        # super().__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

# hyperparamets
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001 #1e11
epochs = 1000



dataset = chatDatset()
train_loader = DataLoader(dataset=dataset, batch_size= batch_size, shuffle = True, num_workers = 2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


### loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.adam(model.parameters(), lr = learning_rate)


for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device) 

        # forwards

        output = model(words)
        loss = loss(output)

        # backprop and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    # every 100 step
    if (epoch+1) % 100 == 0:
        # print('epochs {epoch+1}/{epochs},loss= {loss.item():.4f}')
        print('epochs',epoch+1/epochs)
        print('loss',loss.item())
        
print('Final loss',loss.item())