import copy

# -------- OBJECT CLASSES --------

class Pokemon:
    def __init__(self, name, max_hp):
        self.name, self.nickname = name, name
        self.max_hp, self.hp = max_hp, max_hp
        self.type = None
        self.status = None
        self.ability = None
        self.item = None

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
        

# ------ HELPER FUNCTIONS ------
        
def remove_data_until(log_data, substr):
    while(log_data[0].find(substr) == -1):
        log_data = log_data[1:]
    return log_data


# checks a single line from the log, decides if it is relevant
def line_is_relevant(line):
    if (line.find("|move") > -1 or 
        line.find("|switch") > -1 or 
        line.find("|-damage") > -1 or
        line.find("|faint") > -1):
        return True
    return False


# updates the nickname of a pokemon given its species name, player number
# returns state with the updated nickname
def update_nickname(state, player_num, species, nickname):
    if player_num == 1:
        for pokemon in state.team1:
            if pokemon.name == species:
                pokemon.nickname = nickname
                break
    else:
        for pokemon in state.team2:
            if pokemon.name == species:
                pokemon.nickname = nickname
                break
        
    return state


def lookup_nickname(state, player_num, nickname):
    if player_num == 1:
        for pokemon in state.team1:
            if pokemon.nickname == nickname:
                return pokemon.name
            
    # player_num is 2
    for pokemon in state.team2:
        if pokemon.nickname == nickname:
            return pokemon.name
        
    return "POKEMON NOT FOUND"
    

# returns list of pokemon objects on one team
def get_mons_on_team(log_data, team_name):
    roster = []
    index = 0
    
    while(log_data.find(str("|poke|" + team_name), index) > -1):
        index = log_data.find("|poke|" + team_name, index)
        name_end = log_data.find(",", index+9)
        mon_name = log_data[index+9:name_end]
        
        # non-M/F mons have '|' instead of ',' after name
        if mon_name.find("|") > -1:
            mon_name = mon_name[:mon_name.find("|")]
        
        roster.append(Pokemon(mon_name, 100))
        # move 9 characters past |poke|p#| 
        index += 9 
        
    return roster

# returns player that switched (1 or 2), name and nickname of mon
# modifies log by trimming away data before switch
def parse_from_switch_data(log):
    log = remove_data_until(log, "|switch")
    player = int(log[0][9:10])
    nickname = log[0][13:]
    monname = nickname[nickname.find("|")+1:]
    nickname = nickname[:nickname.find("|")]
    monname = monname[:monname.find(",")]
    if monname.find("|") > -1:
        monname = monname[:monname.find("|")]
    return player, monname, nickname
    

# parameters: 
#   str log: remaining log (will look at first turn in that log)
#   GameState state: current game state object
def parse_one_turn(log, state):
    data_collected = []
    log = remove_data_until(log, "|turn")
    log = log[1:]
    while(len(log) > 0 and log[0].find("|turn") == -1):
        
        if line_is_relevant(log[0]): 
            
            if log[0].find("|move") == 0:
                # parse string
                player = int(log[0][7:8])
                nickname = log[0][11:]
                nickname = nickname[:nickname.find("|")]
                monname = lookup_nickname(state, player, nickname)

                move = log[0][10:]
                move = move[move.find("|")+1:]
                move = move[:move.find("|"):]
                
                # create data point
                d = Decision(True, move)
                state_copy = copy.deepcopy(state)
                data_collected.append(DataPoint(state_copy, d))

            
            elif log[0].find("|switch") == 0:
                # parse string
                player = int(log[0][9:10])
                nickname = log[0][13:]
                monname = nickname[nickname.find("|")+1:]
                nickname = nickname[:nickname.find("|")]
                monname = monname[:monname.find(",")]
                if monname.find("|") > -1:
                    monname = monname[:monname.find("|")]
                
                # update state
                state = update_nickname(state, player, monname, nickname)
                state.cur1 = monname
                
                # create data point
                d = Decision(False, monname)
                state_copy = copy.deepcopy(state)
                data_collected.append(DataPoint(state_copy, d))
                
            elif log[0].find("|-damage") == 0:
                pass
            
            elif log[0].find("|faint") == 0:
                # discard switch that comes from fainting
                log = log[4:]
                
                
        log = log[1:]
    
    return log, state, data_collected


# --------- MAIN DATA COLLECTION PROGRAM --------- 

log_data = data["log"]
log = log_data.split('\n')

# create initial state
team1 = get_mons_on_team(log_data, "p1")
team2 = get_mons_on_team(log_data, "p2")

log = remove_data_until(log, "|start")
log = log[1:]
player1, nickname1, monname1 = parse_from_switch_data(log)
log = log[1:]
player2, nickname2, monname2 = parse_from_switch_data(log)

state = GameState("", "", team1, team2)
state = update_nickname(state, 1, monname1, nickname1)
state = update_nickname(state, 2, monname2, nickname2)

# get data points
data_pts = []
while len(log) > 0:
    log, state, new_data = parse_one_turn(log, state)
    for d in new_data:
        data_pts.append(d)
        
print(data_pts)
