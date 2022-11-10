from html.parser import HTMLParser

import urllib.request
import json

base_url = 'https://replay.pokemonshowdown.com'

# Gather battle urls
with open('battles.csv', 'r') as infile:
    battle_urls = infile.read().split(',')

# Setup for parsing the url
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
header = {'User-Agent': user_agent}

for battle_url in battle_urls:
    # Request data from the url
    request = urllib.request.Request(base_url + battle_url + '.json', headers = header)
    response = urllib.request.urlopen(request)
    battle_webpage_content = response.read()

    # Write the data into a json file
    with open('battle_log/' + battle_url + '.json', 'w') as outfile:
        outfile.write(battle_webpage_content.decode('utf-8'))
