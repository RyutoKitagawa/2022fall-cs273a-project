DataCollection:
	python gatherData/DataCollection.py

e:
	DEBUG="events" python gatherData/DataCollection.py

s:
	DEBUG="switch" python gatherData/DataCollection.py

i:
	DEBUG="init_team" python gatherData/DataCollection.py

neuralNetwork:
	python learningModel/neuralNetwork.py

getBattleData:
	python gaterData/pokemonGatherPlayers.py
	python gaterData/pokemonGatherPlayerData.py
	python gaterData/pokemonGatherBattles.py
