import random
import json
import torch
from model import NeuralNet
from utils import bow, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json') as f:
    intents = json.load(f)




FILE = 'model_state.pth'

data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "gacy"

print("lets chat! type quit to exit")

while True:
    sentense = input('you:')
    if sentense == 'quit':
        break
    tokens = tokenize(sentense)
    bow = bow(sentense, all_words)  
    bow = bow.reshape(1, bow.shape[0])
    bow = torch.from_numpy(bow)

    output = model(bow)
    _,predictions = torch.max(output, dim = 1)
    tag = tags[predictions.items()]

    probs = torch.softmax(output, dime = 1)
    probs = probs[0][predictions.items()]

    if probs.item() > 0.75:
        for intent in intents['intents']:
            if tag == intents['tag']:
                print(bot_name,'',random.choice(intent['responses']))
    else:
        print('I cannot help you with that')
