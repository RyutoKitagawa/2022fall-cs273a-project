# Move Prediction in a 2 Player Incomplete Information Game

## Gathering Data
Running the command below will gather data from the pokemonshowdown.com website, which hosts a database of battles hosted by the website.

```bash
make gatherBattleData
```

Gathering data is split into 3 stages, gathering players, getting all of the player's battle logs, and then saving those battle logs.

### Gathering Players
Gathering players is done with the `gatherData/pokemonGatherPlayers.py` file. Running this file will go to the https://pokemonshowdown.com/ladder/gen8ou website, which hosts the top 500 players in this category, which are saved to the `data/players.csv` file. These players were chosen in part because being on this leaderboard makes it more likely they will be making rational decisions, as well as being more active and providing more data. The "gen8ou" category was chosen since it appears to be the most popular format, and thus will likely provide us with a larger sample size of players, but the "gen8ou" format is only for finding players. When gathering battle logs, we do not restrict ourselves to this format.

### Gathering Player Battle Logs
Gathering the list of battles from each player is done with the `gatherData/pokemonGatherPlayerData.py` file. Each player in the `data/player.csv` file has their own page on the replay.pokemonshowdown.com website containing a list of their battles that have been saved for people to publicly view. Therefore, the webpages containing each of their battles are gathered and stored in a `data/battles.csv` file.

### Saving Battle Logs
Saving each of the battle logs in a json file is done with the `gatherData/pokemonGatherBattles.py` file. The program visits the webpage for every battle in the `data/battles.csv` file, and stores the json file representing that data in the `data/battle_log/` directory. Each of the json files are named after the unique identifier for the website, which includes the specific format of the battle and a string of random numbers as the id of the battle.
