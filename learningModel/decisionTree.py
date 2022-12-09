from sklearn import tree
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

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
    for datapoint in tqdm(datapoints):
        if len(datapoint.split('\n|-s-|\n')) == 3:
            team1, team2, decisions = datapoint.split('\n|-s-|\n')
            data.append(DataPoint(team1, team2, decisions))

print(len(moves) + len(pkms))

training_data_ratio = 0.6
training_data = data[:int(len(data) * training_data_ratio)]
testing_data = data[int(len(data) * training_data_ratio):]

clf = tree.DecisionTreeClassifier()
