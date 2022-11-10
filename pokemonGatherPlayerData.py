from html.parser import HTMLParser

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import urllib.request
import json
import time

class PlayerBattleParsers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.battles = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a' and attrs[0][0] == 'href':
            self.battles.append(attrs[0][1])

# Get list of players from players.csv file
with open('players.csv', 'r') as infile:
    players = infile.read().split(',')

# Open Chrome browser and go to replay page
driver = webdriver.Chrome()
driver.get('https://replay.pokemonshowdown.com')

# Go through each player and gather their battle data
battle_urls = []
for player in players:
    # Refresh the page to reset the page
    driver.refresh()

    # Find the search bar, clear the text, and enter the player's name
    element = driver.find_element('name', 'user')
    element.clear()
    element.send_keys(player)
    element.send_keys(Keys.ENTER)

    # Wait for the page to load
    time.sleep(2)

    # Parse through the HTML to get the battle url
    parser = PlayerBattleParsers()
    parser.feed(driver.page_source)

    # Feed battle url into battle_urls array
    for battle in parser.battles:
        battle_urls.append(battle)

# Write battle urls to the battles.csv file
with open('battles.csv', 'w') as outfile:
    for index, battle in enumerate(battle_urls):
        outfile.write(battle + (','), if index != len(battle_urls) - 1 else '')
