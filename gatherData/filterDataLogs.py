# TODO: Actually write the code (used then deleted earlier)

from os import listdir
import json

for file in listdir('data/battle_log'):
    if (file.endswith('.json')):
        with open('data/battle_log/' + file) as log_file:
            log_data = json.load(log_file)['log']

        log = log_data.split('\n')

        team1, team2 = get_teams(log)

        # Move files that don't have a team to a different folder
        if (len(team1) == 0 or len(team2) == 0):
            print('No team data found in ' + file)
            os.rename('data/battle_log/' + file, 'data/battle_log/no_team/' + file)
