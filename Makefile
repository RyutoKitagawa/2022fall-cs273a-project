all:
	python gatherData/dataObjects.py

test:
	python gatherData/test.py

getBattleData:
	python gaterData/pokemonGatherPlayers.py
	python gaterData/pokemonGatherPlayerData.py
	python gaterData/pokemonGatherBattles.py
