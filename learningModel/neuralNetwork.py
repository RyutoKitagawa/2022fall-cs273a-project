import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm

class Pokemon:
    def __init__(self, index, hp, inplay):
        self.index = index
        self.hp = hp
        self.inplay = inplay

    def __str__(self):
        return f"({pkms[self.index]}, HP: {self.hp}, InPlay: {'T' if self.inplay else 'F'})"

    def __repr__(self):
        return str(self)

class Decison:
    def __init__(self, mos, data):
        self.mos = mos
        self.data = data

    def __str__(self):
        return f"Attack using {moves[self.data]}" if self.mos else f"Switch to {pkms[self.data]}"

    def __repr__(self):
        return str(self)

class DataPoint:
    def __init__(self, team1, team2, decisions):
        self.team1, self.team2, self.decisions = [], [], []

        for pokemon in team1.split('\n'):
            index, hp, inplay = pokemon.split(',')
            self.team1.append(Pokemon(int(index), float(hp), inplay == 'True'))
        
        for pokemon in team2.split('\n'):
            index, hp, inplay = pokemon.split(',')
            self.team2.append(Pokemon(int(index), float(hp), inplay == 'True'))

        for decision in decisions.split('\n'):
            mos, data = decision.split(',')
            self.decisions.append(Decison(mos == 'True', int(data)))

class AttackOrSwitchNN(nn.Module):
    def __init__(self, pkm, moves):
        super(AttackOrSwitchNN, self).__init__()

        inputNum, outputNum = len(pkm) * 4, 2
        self.pkm, self.moves = pkm, moves

        # Input layer represnts the pokemon and their hp and which pokemon is currently in play
        # The first hidden layer is meant to consider the hp of the pokemon of each player's active pokemon
        # The second hidden layer is meant to decide if there is a pokemon that can be switched in
        # The third hidden layer is meant to decide if there is a pokemon that can be switched in
        # This culminates into the output layer which ultimately decides if player 1 and player 2 should attack or switch
        self.fc1 = nn.Linear(inputNum, inputNum)
        self.fc2 = nn.Linear(inputNum, inputNum)
        self.fc3 = nn.Linear(inputNum, inputNum)
        self.fc4 = nn.Linear(inputNum, outputNum)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim = 0)

    def train(self, data):
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        for i in tqdm(range(len(data))):
            optimizer.zero_grad()

            # Get the targets from the data
            target = torch.tensor([float(data[i].decisions[0].mos), float(data[i].decisions[1].mos)])

            try:
                output = self.predict(data[i])
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            except Exception as e:
                pass

    def predict(self, data):
        def gethp(team):
            result = [0] * len(self.pkm)
            for pkm in team:
                result[pkm.index] = pkm.hp
            return result

        getcurpkmidx = lambda team : [i for i in team if i.inplay][0].index
        getcurpkm = lambda idx : [int(idx == i) for i in range(len(self.pkm))]
        team1, team2 = gethp(data.team1), gethp(data.team2)
        team1_curpkm_idx, team2_curpkm_idx = getcurpkmidx(data.team1), getcurpkmidx(data.team2)
        team1_curpkm, team2_curpkm = getcurpkm(team1_curpkm_idx), getcurpkm(team2_curpkm_idx)

        features = torch.tensor(team1 + team2 + team1_curpkm + team2_curpkm)


        output = self.forward(torch.tensor(features))

        return output

    def predict_print(self, data):
        output = self.predict(data)

        print('Player 1:', 'Attack' if output[0] > 0.5 else 'Switch')
        print('Player 2:', 'Attack' if output[1] > 0.5 else 'Switch')

    def calculate_error(self, data):
        correct, total = 0, len(data) * 2
        
        for d in tqdm(data):
            try:
                output = self.predict(d)

                if output[0] > 0.5 and d.decisions[0].mos:
                    correct += 1
                elif output[0] < 0.5 and not d.decisions[0].mos:
                    correct += 1

                if output[1] > 0.5 and d.decisions[1].mos:
                    correct += 1
                elif output[1] < 0.5 and not d.decisions[1].mos:
                    correct += 1

            except Exception as e:
                total -= 1

        return correct / total

class PkmNN(nn.Module):
    def __init__(self, pkm, moves):
        super(PkmNN, self).__init__()

        inputNum, outputNum = len(pkm) * 4, 2 * len(moves) + 2 * len(pkm)
        self.pkm, self.moves = pkm, moves

        self.fc1 = nn.Linear(inputNum, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, outputNum)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def train(self, features, targets):
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        for i in tqdm(range(len(features))):
            optimizer.zero_grad()

            output = self.predict(features[i])
            targets[i] = torch.tensor([float(k) for j in targets[i] for k in j.split(',')[:-1]])
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

print('Loading Pokemon Data ...')
with open('data/unique_pokemon.csv', 'r') as f:
    pkms = f.read().split(',')

print('Loading Move Data ...')
with open('data/unique_moves.csv', 'r') as f:
    moves = f.read().split(',')

print('Loading Training Data ...')
with open('data/datapoints.csv', 'r') as f:
    print('Splitting Data ...')
    data, datapoints = [], f.read().split('\n|-o-|\n')
    flag = True
    for datapoint in tqdm(datapoints):
        if len(datapoint.split('\n|-s-|\n')) == 3:
            team1, team2, decisions = datapoint.split('\n|-s-|\n')
            data.append(DataPoint(team1, team2, decisions))

training_data_ratio = 0.6
training_data = data[:int(len(data) * training_data_ratio)]
testing_data = data[int(len(data) * training_data_ratio):]

#net = PkmNN(pkms, moves)
net = AttackOrSwitchNN(pkms, moves)
print('Training ...')
net.train(training_data)
print('Calculating Error ...')
print(net.calculate_error(training_data))
print(net.calculate_error(testing_data))
net.predict_print(data[0])
