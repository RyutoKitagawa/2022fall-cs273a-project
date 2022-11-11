all:
	python gatherData/dataObjects.py

getBattleData:
	python gaterData/pokemonGatherPlayers.py
	python gaterData/pokemonGatherPlayerData.py
	python gaterData/pokemonGatherBattles.py
