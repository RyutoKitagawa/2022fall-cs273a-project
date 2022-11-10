with open('battles.csv', 'r') as infile:
    battle_urls = infile.read().split(',')

print(len(battle_urls))
