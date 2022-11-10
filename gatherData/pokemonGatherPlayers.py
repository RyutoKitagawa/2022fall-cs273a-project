from html.parser import HTMLParser

import urllib.request
import json

class PokemonHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.players = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            playerNames = attrs[-1][-1]
            self.players.append(playerNames[7:])

# Setup for parsing the url
ladder_url = 'https://pokemonshowdown.com/ladder/gen8ou'
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
header = {'User-Agent': user_agent}

# Request data from the url
request = urllib.request.Request(ladder_url, headers = header)
response = urllib.request.urlopen(request)
ladder_webpage_content = response.read()

# Parse through the data from the url
parser = PokemonHTMLParser()
parser.feed(ladder_webpage_content.decode('utf-8'))

# Output parsed data of players to a CSV file
with open('data/players.csv', 'w') as outfile:
    for index, player in enumerate(parser.players[7:]):
        outfile.write(player + (',' if index < len(parser.players[7:]) - 1 else ''))
