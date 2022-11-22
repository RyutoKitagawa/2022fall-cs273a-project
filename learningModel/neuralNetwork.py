import tensorflow as tf 

class PokemonPredictorNeuralNetwork:
    def __init__(self, pokemon, moves):
        # Each tema input is an array of 978 numbers
        # Each number represents the amount of health a pokemon has
        # A pokemon with 0 health is either not on the team or fainted
        # In either case, the pokemon is inconsequential to the battle
        self.input_team1 = tf.keras.models.Input(shape=(len(pokemon),))
        self.input_team2 = tf.keras.models.Input(shape=(len(pokemon),))

        # TODO: Create a list of hidden layers with logic behind it
        self.hidden_layers = [tf.keras.layers.Dense(len(pokemon) // i, activation='relu') for _ in range(5)]

        self.output_layer = tf.keras.layers.
