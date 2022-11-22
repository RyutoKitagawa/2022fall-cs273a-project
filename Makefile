DataCollection:
	#DEBUG="switch" python gatherData/DataCollection.py
	#DEBUG="events" python gatherData/DataCollection.py
	#DEBUG="init_team" python gatherData/DataCollection.py
	python gatherData/DataCollection.py

neuralNetwork:
	python learningModel/neuralNetwork.py

getBattleData:
	python gaterData/pokemonGatherPlayers.py
	python gaterData/pokemonGatherPlayerData.py
	python gaterData/pokemonGatherBattles.py
