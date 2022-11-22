# TODO: Actually write the code (used then deleted earlier)

from os import listdir
import json

import pokemonDataObject
from pokemonDataObject import Pokemon

# Iterate through each of the files in the data/battle_log/ directory
# Filters out undesirable data: empty files and files with pokmeon not listed in the beginning
def filterDataLogs():
    for file in listdir('data/battle_log'):
        try:
            with open('data/battle_log/' + file) as log_file:
                log_data = json.load(log_file)['log']

            log = log_data.split('\n')

            team1, team2 = pokemonDataObject.get_teams(log)

            # Move files that don't have a team to the no_team folder
            if (len(team1) == 0 or len(team2) == 0):
                print('No team data found in ' + file)
                os.rename('data/battle_log/' + file, 'data/no_team/' + file)
        except:
            os.rename('data/battle_log/' + file, 'data/not_connected/' + file)

def gatherUniquePokemon():
    unique_pokemon = set()

    for file in listdir('data/battle_log'):
        with open('data/battle_log/' + file) as log_file:
            log_data = json.load(log_file)['log']

        log = log_data.split('\n')

        team1, team2 = pokemonDataObject.get_teams(log)

        for pokemon in team1:
            unique_pokemon.add(pokemon.name)

        for pokemon in team2:
            unique_pokemon.add(pokemon.name)

    return unique_pokemon

def outputFileOfUniquePokemon():
    pokemon = gatherUniquePokemon()
    with open('data/unique_pokemon.csv', 'w') as file:
        for pokemon_name in list(pokemon)[:-1]:
            file.write(pokemon_name + ',')
        file.write(list(pokemon)[-1])
