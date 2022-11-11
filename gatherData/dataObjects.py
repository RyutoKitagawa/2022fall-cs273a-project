import json
from os import listdir

class Pokemon:
    def __init__(self, name, max_hp = 100):
        self.name, self.nickname = name, name
        self.max_hp, self.hp = max_hp, max_hp
        self.type = None
        self.status = None
        self.ability = None
        self.item = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class GameState:
    # records a game state 
    # cur: current mon in play, team: remaining team list
    def __init__(self, cur1, cur2, team1, team2):
        self.cur1, self.cur2 = cur1, cur2
        self.team1, self.team2 = team1, team2
        
class Decision:
    # records a decision made by a player
    # mos: boolean, true if attack/move, false if switch
    # move_data: string, name of mon switched to or move used
    def __init__(self, mos, move_data):
        self.mos = mos
        self.move_data = move_data
        
class DataPoint:
    # records a game state and a decision a player made
    def __init__(self, state, decision):
        self.state, self.decision = state, decision
        
def remove_data_until(log_data, substr):
    while (log_data[0].find(substr) == -1):
        log_data.pop(0)

def find_commands(log_data, command):
    commands = []
    for i in log_data:
        if (i.find(command) != -1):
            commands.append(i)

    return commands

def get_teams(log_data):
    team1, team2 = [], []
    team_data = find_commands(log_data, '|poke|p')

    for i in team_data:
        data = i.split('|')[2:]

        if (data[0] == 'p1'):
            team1.append(Pokemon(data[1].split(',')[0]))
        elif (data[0] == 'p2'):
            team2.append(Pokemon(data[1].split(',')[0]))
        else:
            printf('Error: invalid team data')
            exit(1)

    return team1, team2

count = 0
for file in listdir('data/battle_log'):
    try:
        with open('data/battle_log/' + file) as log_file:
            log_data = json.load(log_file)['log']

        log = log_data.split('\n')

        team1, team2 = get_teams(log)

    except:
        pass
