# log data from json object is given
log_data = data["log"]

# parse out pokemon on teams

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
        
        roster.append(mon_name)
        # move 9 characters past |poke|p#| 
        index += 9 
        
    return roster

team1 = get_mons_on_team(log_data, 'p1')
team2 = get_mons_on_team(log_data, 'p2')
print(team1)
print(team2)
              
  
class GameState:
    # records a game state; cur: current mon in play, 
    # team: remaining team list, hp: remaining % on current mon
    def __init__(self, cur1, cur2, team1, team2, hp1, hp2):
        self.cur1 = cur1
        self.cur2 = cur2
        self.team1 = team1
        self.team2 = team2
        self.hp1 = hp1
        self.hp2 = hp2
        
class Decision:
    # records a game state and what decisions each player made
    # aos: boolean, true if attack/move, false if switch
    # movedata: string, name of mon switched to or move used
    def __init__(self, state, aos1, aos2, movedata1, movedata2):
        self.state = state
        self.aos1 = aos1
        self.aos2 = aos2
        self.movedata1 = movedata1
        self.movedata2 = movedata2
        
# gets the number of given mon for hp tracking purposes
def get_mon_num(mon_name, team):
    for i in range(len(team)):
        if team[i] == mon_name:
            return i
    return -1

        
# record current pokemon in play and hp pcts
team1 = get_mons_on_team(log_data, 'p1')
team2 = get_mons_on_team(log_data, 'p2')
hp1 = [1]*len(team1)
hp2 = [1]*len(team2)


cur1 = log_data[log_data.find("|switch|p1a:")+12:]
cur2 = log_data[log_data.find("|switch|p2a:")+12:]
cur1 = cur1[cur1.find("|")+1:cur1.find(",")]
cur2 = cur2[cur2.find("|")+1:cur2.find(",")]

if cur1.find("|") > -1:
    cur1 = cur1[:cur1.find("|")]
if cur2.find("|") > -1:
    cur2 = cur2[:cur2.find("|")]

cur1num = get_mon_num(cur1, team1)
cur2num = get_mon_num(cur2, team2)

print(cur1)
print(cur2)
print(cur1num)
print(cur2num)

my_data = []
        
#while log_data.find("|switch|p1a:") > -1 or log_data.find("|switch|p1a:") > -1:
    
    
