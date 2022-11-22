all: neuralNetwork

test:
	python gatherData/test.py

neuralNetwork:
	python learningModel/neuralNetwork.py

getBattleData:
	python gaterData/pokemonGatherPlayers.py
	python gaterData/pokemonGatherPlayerData.py
	python gaterData/pokemonGatherBattles.py
