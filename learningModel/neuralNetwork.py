import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class PkmNN(nn.Module):
    def __init__(self, pkm, moves):
        super(PkmNN, self).__init__()

        inputNum = len(pkm) * 4
        outputNum = len(moves) + len(pkm) + len(moves) + len(pkm)

        self.fc1 = nn.Linear(inputNum, 5)
        self.fc2 = nn.Linear(5, 2)
        self.fc3 = nn.Linear(2, outputNum)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def train(self, features, targets):
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        for i in range(len(features)):
            optimizer.zero_grad()

            output = self.predict(features[i])
            targets[i] = torch.tensor([float(j) for i in targets[i] for j in i.split(',')[:-1]])
            loss = F.binary_cross_entropy(output, targets[i])
            loss.backward()
            optimizer.step()

    def predict(self, features):
        features = [float(j) for i in features for j in i.split(',')[:-1]]
        output = self.forward(torch.tensor(features))

        return output

    def predict_print(self, features):
        features = [float(j) for i in features for j in i.split(',')[:-1]]
        output = self.forward(torch.tensor(features))

        print('Team 1 Decision:', key[torch.argmax(output[:len(key)])])
        print('Team 2 Decision:', key[torch.argmax(output[len(key):])])

def prettyPrintFeatures(features):
    print('Team 1:')
    for i, ele in enumerate(features[0].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{pkms[i]}({ele}),', end=' ')
    print()

    print('Team 2:')
    for i, ele in enumerate(features[1].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{pkms[i]}({ele}),', end=' ')
    print()

    print('Team 1 Pokemon Currently In Play: ', end='')
    for i, ele in enumerate(features[2].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{pkms[i]}')

    print('Team 2 Pokemon Currently In Play: ', end='')
    for i, ele in enumerate(features[3].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{pkms[i]}')

def prettyPrintTarget(target):
    print('Team 1 Decision:', end = '')
    for i, ele in enumerate(target[0].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{moves[i]}', end='')

    for i, ele in enumerate(target[1].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{pkms[i]}', end='')

    print()

    print('Team 2 Decision:', end = '')
    for i, ele in enumerate(target[2].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{moves[i]}', end='')

    for i, ele in enumerate(target[3].split(',')[:-1]):
        if float(ele) > 0:
            print(f'{pkms[i]}', end='')

    print()

with open('data/unique_pokemon.csv', 'r') as f:
    pkms = f.read().split(',')

with open('data/moves.csv', 'r') as f:
    moves = f.read().split(',')

with open('data/datapoints.csv', 'r') as f:
    data = f.read().split('\n')[:-1]
    features = [data[i:i + 4] for i in range(0, len(data), 8)]
    targets = [data[i + 4:i + 8] for i in range(0, len(data), 8)]

key = pkms + moves

net = PkmNN(pkms, moves)
net.train(features, targets)
net.predict_print(features[0])
