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
        
