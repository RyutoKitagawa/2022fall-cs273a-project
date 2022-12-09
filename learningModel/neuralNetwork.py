import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm

import random as rand

from sklearn import metrics

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

class NeuralNetwork(nn.Module):
    def __init__(self, pkm, moves):
        super(NeuralNetwork, self).__init__()

    def bulk_predict(self, data):
        result = []
        for i, d in enumerate(tqdm(data)):
            try:
                result.append((i, self.predict(d)))
            except Exception as e:
                pass

        return result

    def train(self, data):
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        epoch, batch_size = 100, 3000
        indexes = [i for i in range(len(data))]

        for _ in tqdm(range(epoch)):
            rand.shuffle(indexes)

            for i in indexes[:batch_size]:
                optimizer.zero_grad()

                # Get the targets from the data
                target = self.get_target(data[i])

                try:
                    output = self.predict(data[i])
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

                    if loss < 0.1:
                        break

                except Exception as e:
                    pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

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

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        return F.softmax(x, dim = 0)


# Class that inherits from the NeuralNetwork and 
class AttackOrSwitchNN(NeuralNetwork):
    def __init__(self, pkm, moves):
        super().__init__(pkm, moves)

        inputNum, outputNum = len(pkm) * 4, 2
        self.pkm, self.moves = pkm, moves

        # Input layer represnts the pokemon and their hp and which pokemon is currently in play
        # The first hidden layer is meant to consider the hp of the pokemon of each player's active pokemon
        # The second hidden layer is meant to decide if there is a pokemon that can be switched in
        # The third hidden layer is meant to decide if there is a pokemon that can be switched in
        # This culminates into the output layer which ultimately decides if player 1 and player 2 should attack or switch
        layerSizes = [inputNum, (inputNum + outputNum) // 2, outputNum]

        self.layers = nn.ModuleList()
        for i in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[i], layerSizes[i + 1]))

    def get_target(self, data):
        return torch.tensor([float(data.decisions[0].mos), float(data.decisions[1].mos)])

    def predict_print(self, data):
        output = self.predict(data)

        print('Player 1:', 'Attack' if output[0] > 0.5 else 'Switch')
        print('Player 2:', 'Attack' if output[1] > 0.5 else 'Switch')

    def calculate_error(self, data):
        correct, total = 0, len(data) * 2
        
        for d in tqdm(data):
            try:
                output = self.predict(d)

                correct += int((output[0] > 0.5 and d.decisions[0].mos) or (output[0] < 0.5 and not d.decisions[0].mos))
                correct += int((output[1] > 0.5 and d.decisions[1].mos) or (output[1] < 0.5 and not d.decisions[1].mos))

            except Exception as e:
                total -= 1

        return correct / total

class PkmNN(NeuralNetwork):
    def __init__(self, pkm, moves):
        super().__init__(pkm, moves)

        inputNum, outputNum = len(pkm) * 4, 2 * len(moves) + 2 * len(pkm)
        self.pkm, self.moves = pkm, moves

        layerSizes = [inputNum, (inputNum + outputNum) // 2, outputNum]
        self.layers = nn.ModuleList()
        for i in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[i], layerSizes[i + 1]))

    def predict_print(self, data):
        output = self.predict(data)

        decision1 = torch.argmax(output[:len(output) // 2])
        decision2 = torch.argmax(output[len(output) // 2:])

        print('Player 1: ', end = '')
        if decision1 >= len(pkms):
            print(f'Attack with {moves[decision1 - len(pkms)]}')
        else:
            print(f'Switch to {pkms[decision1]}')

        print('Player 2: ', end = '')
        if decision2 >= len(pkms):
            print(f'Attack with {moves[decision2 - len(pkms)]}')
        else:
            print(f'Switch to {pkms[decision2]}')

    def get_target(self, data):
        target = torch.tensor([0] * (2 * len(pkms) + 2 * len(moves)))
        if not data.decisions[0].mos:
            target[data.decisions[0].data] = 1
        else:
            target[len(pkms) + data.decisions[0].data] = 1

        if not data.decisions[1].mos:
            target[len(pkms) + len(moves) + data.decisions[1].data] = 1
        else:
            target[2 * len(pkms) + len(moves) + data.decisions[1].data] = 1

        return target

    def calculate_error(self, data):
        correct, total = 0, len(data) * 2
        
        for d in tqdm(data):
            try:
                output = self.predict(d)
                decision1 = torch.argmax(output[:len(output) // 2])
                decision2 = torch.argmax(output[len(output) // 2:])

                if not d.decisions[0].mos:
                    correct += int(d.decisions[0].data == decision1)
                else:
                    correct += int(d.decisions[0].data == decision1 - len(pkms))

                if not d.decisions[1].mos:
                    correct += int(d.decisions[1].data == decision1)
                else:
                    correct += int(d.decisions[1].data == decision1 - len(pkms))

            except Exception as e:
                total -= 1

        return (total - correct) / total


print('Loading Pokemon Data ...')
with open('data/unique_pokemon.csv', 'r') as f:
    pkms = f.read().split(',')

print('Loading Move Data ...')
with open('data/unique_moves.csv', 'r') as f:
    moves = f.read().split(',')

print('Loading Training Data ...')
with open('data/datapoints.csv', 'r') as f:
    count = 0
    print('Splitting Data ...')
    data, datapoints = [], f.read().split('\n|-o-|\n')
    for datapoint in tqdm(datapoints):
        if len(datapoint.split('\n|-s-|\n')) == 3:
            team1, team2, decisions = datapoint.split('\n|-s-|\n')
            data.append(DataPoint(team1, team2, decisions))

training_data_ratio = 0.6
training_data = data[:int(len(data) * training_data_ratio)]
testing_data = data[int(len(data) * training_data_ratio):]

net = AttackOrSwitchNN(pkms, moves)
#net = PkmNN(pkms, moves)
#print('Training ...')
#net.train(training_data)
#net.save('models/TrueChoice.model')
net.load('models/attackSwitch.model')
#print('Calculating Error ...')
#print('Training Error Rate:', net.calculate_error(training_data))
#print('Test Error Rate:', net.calculate_error(testing_data))
net.predict_print(data[0])
prediction_full = net.bulk_predict(testing_data)
true_predictions = [int(j.mos) for i in prediction_full for j in data[i[0]].decisions]
predictions = [j for i in prediction_full for j in i[1]]
error = sum([int(abs(i - j) > 0.5) for i, j in zip(true_predictions, predictions)]) / len(true_predictions)

fpr, tpr, thresholds = metrics.roc_curve(true_predictions, predictions)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
