import copy
import json
import os
from debugprint import Debug
from tqdm import tqdm


# -------- OBJECT CLASSES --------

status = set(['brn', 'par', 'slp', 'frz', 'psn', 'tox'])

debug_switch = Debug('switch')
debug_init_team = Debug('init_team')
debug_events = Debug('events')

class Team:
    # The team is a list of Pokemon objects
    # where the objects are created through looking
    # at the log file.
    def __init__(self, log, team_name):
        pkms = [i.split('|')[3].split(',')[0] for i in log if i.startswith('|poke|' + team_name)]
        pkms = [i.split('-')[0] for i in pkms]
        self.roster = {pkm: Pokemon(pkm) for pkm in pkms}

        debug_init_team(self.roster, 'roster')

        nickname, pkmName = [i.split('|')[2:4] for i in log if i.startswith('|switch|' + team_name)][0]
        nickname, pkmName = nickname.split(': ')[1], pkmName.split(',')[0]

        self.switch(pkmName, nickname)

    # When a Pokemon switches out, this function can be called to update
    # team's current pokemon, as well as updating their nickname since the
    # nicknames are only revealed when they switch in
    def switch(self, pkmName, nickname, hpratio = None):
        debug_switch(pkmName, 'name')
        debug_switch(nickname, 'nickname')
        debug_switch(self.roster, 'roster')

        if len(pkmName.split('-')) > 1:
            pkmName = pkmName.split('-')[0]

        if len(nickname.split('-')) > 1:
            nickname = nickname.split('-')[0]

        if nickname not in self.roster and pkmName in self.roster:
            self.roster[nickname] = self.roster.pop(pkmName)
            self.roster[nickname].nickname = nickname

        self.currPkm = self.roster[nickname].name
        self.calculate_damage(nickname, hpratio)

    def calculate_damage(self, target, hpratio):
        if hpratio is None:
            return

        if len(target.split('-')) > 1:
            target = target.split('-')[0]

        if len(hpratio) < 2:
            hpratio = ['0', '100']

        if hpratio[-1][-3:] in status:
            hpratio[-1] = hpratio[-1][:-4]

        if hpratio[-1][-1] == 'g':
            hpratio[-1] = hpratio[-1][:-1]

        self.roster[target].hp = int(hpratio[0]) / int(hpratio[1])

    def __str__(self):
        return 'Curently out: ' + str(self.currPkm) + '\n' + str([pkm for pkm in self.roster.values()])

    def search(self, name):
        team = [i for i in self.roster.values()]
        for pkm in team:
            if name in pkm.name:
                return pkm

        return None

class Pokemon:
    # We calculate the hp through percentages
    def __init__(self, name):
        self.name, self.nickname = name, name
        self.hp = 1.0
        self.type = None
        self.status = None
        self.ability = None
        self.item = None

    def __str__(self):
        return f'{self.name} ({self.nickname}): {self.hp * 100}/100 HP'

    def __repr__(self):
        return str(self)

class Decision:
    # records a decision made by a player
    # mos: boolean, true if attack/move, false if switch
    # move_data: string, name of mon switched to or move used
    def __init__(self, mos, move_data):
        self.mos = mos
        self.move_data = move_data

    def __str__(self):
        return f"(Decision: {self.mos}, {self.move_data})"

    def __repr__(self):
        return str(self)
        
class DataPoint:
    # records a game state and a decision a player made
    def __init__(self, team1, team2, log):
        self.teams = {'p1': copy.deepcopy(team1), 'p2': copy.deepcopy(team2)}
        self.new_teams = {'p1': copy.deepcopy(team1), 'p2': copy.deepcopy(team2)}

        self.p1_decision = self.process_turn(log, 'p1')
        self.p2_decision = self.process_turn(log, 'p2')

    def __str__(self):
        return 'Player 1 Team\n---------\n' + str(self.teams['p1']) + '\nPlayer 2 Team\n---------\n' + str(self.teams['p2']) + '\nPlayer 1 Decision: ' + str(self.p1_decision) + '\nPlayer 2 Decision: ' + str(self.p2_decision) + '\n'

    def __repr__(self):
        return str(self)

    def process_turn(self, log, player):
        events = [i.split('|')[1:] for i in log if len(i.split('|')) > 2 and i.split('|')[2].startswith(player)]

        decision = []
        for event in events:
            debug_events(event)
            if event[0] == 'move':
                decision.append(Decision(True, event[2]))
            elif event[0] == 'switch':
                pkmName, nickname, hpratio = event[2].split(',')[0], event[1].split(': ')[1], event[3].split('/')
                decision.append(Decision(False, pkmName.split('-')[0]))
                self.new_teams[player].switch(pkmName, nickname, hpratio)
            elif event[0] == '-damage':
                nickname, hpratio = event[1].split(': ')[1], event[2].split(' ')[0].split('/')
                self.new_teams[player].calculate_damage(nickname, hpratio)
            elif event[0] == 'drag':
                pkmName, nickname, hpratio = event[2].split(',')[0], event[1].split(': ')[1], event[3].split('/')
                self.new_teams[player].switch(pkmName, nickname, hpratio)
            elif event[0] == 'faint':
                nickname, hpratio = event[1].split(': ')[1], ['0', '100']
                self.new_teams[player].calculate_damage(nickname, hpratio)

        return decision

# ------ HELPER FUNCTIONS ------

def shouldSkip(turns):
    if len(turns) < 2:
        return True

    if '|gametype|doubles' in turns[0]:
        return True

    rules = [i.split('|')[2].split(':')[0] for i in turns[0] if i.startswith('|rule|')]
    
    if 'Species Clause' not in rules:
        return True

    return False

# --------- MAIN DATA COLLECTION PROGRAM --------- 
battles = os.listdir('data/battle_log')
datapoints = []

pkms, moves = set(), set()

print('Collecting data...')
for battle in tqdm(battles):
    with open (f'data/battle_log/{battle}') as f:
        data = json.load(f)

    log_data = data["log"]
    turns = [i.split('\n') for i in log_data.split("|turn|")]

    if shouldSkip(turns):
        continue

    team1 = Team(turns[0], "p1")
    team2 = Team(turns[0], "p2")

    if 'Zoroark' in team1.roster or 'Zoroark' in team2.roster:
        continue

    for turn in turns[1:]:
        datapoints.append(DataPoint(team1, team2, turn))
        team1, team2 = datapoints[-1].new_teams['p1'], datapoints[-1].new_teams['p2']

        for pkm in team1.roster.values():
            pkms.add(pkm.name)

        for pkm in team2.roster.values():
            pkms.add(pkm.name)

        for decision in datapoints[-1].p1_decision:
            if decision.mos:
                moves.add(decision.move_data)

        for decision in datapoints[-1].p2_decision:
            if decision.mos:
                moves.add(decision.move_data)

pkms, moves = list(pkms), list(moves)
key = {**{name: i for i, name in enumerate(pkms)}, **{name: i for i, name in enumerate(moves)}}

print('Write pokemon data to file...')
with open('data/unique_pokemon.csv', 'w') as f:
    f.write(','.join(pkms))

print('Write move data to file...')
with open('data/unique_moves.csv', 'w') as f:
    f.write(','.join(moves))

with open(f'data/datapoints.csv', 'w') as f:
    for datapoint in tqdm(datapoints):
        if len(datapoint.p1_decision) and len(datapoint.p2_decision):
            for pkm in datapoint.teams['p1'].roster:
                f.write(f"{key[datapoint.teams['p1'].roster[pkm].name]}")
                f.write(f",{datapoint.teams['p1'].roster[pkm].hp}")
                f.write(f",{datapoint.teams['p1'].currPkm == pkm}\n")
            f.write('|-s-|\n')

            for pkm in datapoint.teams['p2'].roster:
                f.write(f"{key[datapoint.teams['p2'].roster[pkm].name]}")
                f.write(f",{datapoint.teams['p2'].roster[pkm].hp}")
                f.write(f",{datapoint.teams['p2'].currPkm == pkm}\n")
            f.write('|-s-|\n')

            f.write(f'{datapoint.p1_decision[0].mos},{key[datapoint.p1_decision[0].move_data]}\n')
            f.write(f'{datapoint.p2_decision[0].mos},{key[datapoint.p2_decision[0].move_data]}\n')

            f.write('|-o-|\n')
