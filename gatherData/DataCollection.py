import copy
import json

# -------- OBJECT CLASSES --------

class Team:
    # The team is a list of Pokemon objects
    # where the objects are created through looking
    # at the log file.
    def __init__(self, log, team_name):
        pkms = [i.split('|')[-2].split(',')[0] for i in log if i.startswith('|poke|' + team_name)]
        self.roster = {pkm: Pokemon(pkm) for pkm in pkms}

        nickname, pkmName = [i.split('|')[2:4] for i in log if i.startswith('|switch|' + team_name)][0]
        nickname, pkmName = nickname.split(': ')[1], pkmName.split(',')[0]

        self.switch(pkmName, nickname)

    # When a Pokemon switches out, this function can be called to update
    # team's current pokemon, as well as updating their nickname since the
    # nicknames are only revealed when they switch in
    def switch(self, pkmName, nickname):
        if nickname not in self.roster and pkmName in self.roster:
            self.roster[nickname] = self.roster.pop(pkmName)
            self.roster[nickname].nickname = nickname

        self.currPkm = self.roster[nickname].name

    def calculate_damage(self, target, hpratio):
        self.roster[target].hp = int(hpratio[0]) / int(hpratio[1])

    def __str__(self):
        return 'Curently out: ' + str(self.currPkm) + '\n' + str([pkm for pkm in self.roster.values()])

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
        return f"Decision: {self.mos}, {self.move_data}"
        
class DataPoint:
    # records a game state and a decision a player made
    def __init__(self, team1, team2, log):
        self.teams = {'p1': copy.deepcopy(team1), 'p2': copy.deepcopy(team2)}
        self.p1_decision = self.process_turn(log, 'p1')
        self.p2_decision = self.process_turn(log, 'p2')

    def __str__(self):
        return str(self.teams['p1']) + '\n\n' + str(self.teams['p2']) + '\nPlayer 1: ' + str(self.p1_decision) + '\nPlayer 2: ' + str(self.p2_decision)

    def process_turn(self, log, player):
        events = [i.split('|')[1:] for i in log if len(i.split('|')) > 2 and i.split('|')[2].startswith(player)]

        decision = None
        for event in events:
            if event[0] == 'move':
                decision = Decision(True, event[2])
            elif event[0] == 'switch':
                pkmName, nickname = event[2].split(',')[0], event[1].split(': ')[1]
                decision = Decision(False, nickname)
                self.teams[player].switch(pkmName, nickname)
            elif event[0] == '-damage':
                nickname, hpratio = event[1].split(': ')[1], event[2].split(' ')[0].split('/')
                self.teams[player].calculate_damage(nickname, hpratio if len(hpratio) == 2 else [0, 1])

        return decision

# ------ HELPER FUNCTIONS ------
        
# checks a single line from the log, decides if it is relevant
def line_is_relevant(line):
    if (line.find("|move") > -1 or 
        line.find("|switch") > -1 or 
        line.find("|-damage") > -1 or
        line.find("|faint") > -1):
        return True
    return False

# --------- MAIN DATA COLLECTION PROGRAM --------- 
#with open ('data/battle_log/gen8ou-1675763469.json') as f:
with open ('data/example_data.json') as f:
    data = json.load(f)

log_data = data["log"]
turns = [i.split('\n') for i in log_data.split("|turn|")]

# create initial state
team1 = Team(turns[0], "p1")
team2 = Team(turns[0], "p2")

datapoints = []
for turn in turns[1:]:
    datapoints.append(DataPoint(team1, team2, turn))
    team1, team2 = datapoints[-1].teams['p1'], datapoints[-1].teams['p2']
    print(datapoints[-1])
