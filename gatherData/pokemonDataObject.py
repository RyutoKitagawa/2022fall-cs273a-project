class Pokemon:
    def __init__(self, name, max_hp = 100):
        self.name, self.nickname = name, name
        self.max_hp, self.hp = max_hp, max_hp
        self.mega_evolve, self.dynamax = False, False
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

# Find all of the lines in the log that contains the command
def find_commands(log_data, command):
    commands = []
    for i in log_data:
        if (i[:len(command)] == command):
            commands.append(i)

    return commands

# Within a single data log, parse out the Pokemon in each team and return them
# The json file must contain a list of the pokemon on each team at the beginning of the game
def get_teams(log_data):
    team1, team2 = [], []

    # Find each line that contains the pokemon for each team
    team_data = find_commands(log_data, '|poke|p')

    for i in team_data:
        # Splits the data by | and then removes the first two elements
        # The first element will always be empty, since each line starts with a |
        # The second element will always be 'poke', since that is the command we searched for
        # Neither of these are necessary for the rest of the function
        data = i.split('|')[2:]

        # The first element tells us which team the pokemon is on
        # The second element is the name of the pokemon as well as some additional information that we are currently ignoring
        if (data[0] == 'p1'):
            team1.append(Pokemon(data[1].split(',')[0]))
        elif (data[0] == 'p2'):
            team2.append(Pokemon(data[1].split(',')[0]))
        else:
            # In the case that something about the data is screwed up
            printf('Error: invalid team data')
            exit(1)

    return team1, team2
