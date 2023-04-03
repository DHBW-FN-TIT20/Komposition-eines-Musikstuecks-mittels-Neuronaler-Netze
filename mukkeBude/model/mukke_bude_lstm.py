from typing import Any

import keras
import numpy as np
from mapping import MusicMapping


class MukkeBudeLSTM:
    def __init__(
        self,
        mapping: MusicMapping,
        hidden_layer: list[int] = [256],
        loss="sparse_categorical_crossentropy",
        activation="softmax",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        sequence_length: int = 64,
    ) -> None:
        """LSTM model for MukkeBude

        :param mapping: Dictionary mapping unique symbols to integers
        :param hidden_layer: Number of hidden Layers, defaults to [256]
        :param loss: Loss function of the LSTM, defaults to "sparse_categorical_crossentropy"
        :param activation: Activation function of the LSTM, defaults to "softmax"
        :param optimizer: optemizer of the LSTM, defaults to keras.optimizers.Adam(learning_rate= 0.001)
        :param sequence_length: Length of the sequences, defaults to 64
        """
        self.mapping = mapping
        self.output_layer_size = len(mapping)
        self.hidden_layer_sizes = hidden_layer
        self.loss = loss
        self.activation = activation
        self.optimizer = optimizer
        self.sequence_length = sequence_length

        # Input layer
        input_layer = keras.layers.Input(shape=(None, self.output_layer_size))

        # Hidden layers
        x = input_layer
        for hidden_layer_size in self.hidden_layer_sizes:
            x = keras.layers.LSTM(hidden_layer_size)(x)

        # Avoid overfitting
        x = keras.layers.Dropout(0.3)(x)

        # Output layer. Full connection -> Dense
        output_layer = keras.layers.Dense(self.output_layer_size, activation=self.activation)(x)

        # The actual model
        self.model = keras.Model(input_layer, output_layer)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])

    def train(self, dataset: list[int], epochs: int = 50, batch_size: int = 64) -> None:
        """Train the LSTM model

        :param dataset: Training dataset
        :param epochs: Number of epochs to train, defaults to 10
        :param batch_size: Size of the batches, defaults to 64
        """
        inputs, targets = self.__create_training_data(dataset)
        self.model.fit(inputs, targets, epochs=epochs, batch_size=batch_size)

    def generate(
        self,
        start_seed: str,
        max_length: int = 128,
        max_sequence_length: int = 64,
        temperature=0.8,
    ) -> list[str]:
        """Generate a new song

        :param start_seed: Starting string with start symbols and notes
        :param max_length: Maximum length of the generated song, defaults to 128
        :param temperature: temperature of the next character choice, defaults to 0.8
        :return: Generated song
        """

        # map the ssed to integers
        seed = self.mapping.numericalize(start_seed.split(" "))

        # Start the melody generation
        output_melody = []
        for _ in range(max_length):
            # limit the seed to max_squenece_length
            # cut off the beginning of the seed if it is longer than max_sequence_length
            seed = seed[-max_sequence_length:]

            # One-Hot the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self.mapping))

            # Reshape the seed so that it can be fed to the model
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Make a prediction
            probabilities = self.model.predict(onehot_seed)[0]

            # Apply the temperature to the probabilities
            predictions = np.log(probabilities) / temperature
            probabilities = np.exp(predictions) / np.sum(
                np.exp(predictions),
            )  # normalize the probabilities. This is like applying softmax

            # Get the index of the predicted note
            index = np.random.choice(len(self.mapping), p=probabilities)

            # Update the seed
            seed.append(index)

            # Map the index to the note
            output_symbol = self.mapping.textify(
                [
                    index,
                ],
            )

            # Stop if we have reached the end of the melody
            if output_symbol == "/":
                break

            # Update the output melody
            output_melody.append(output_symbol)

        return output_melody

    def __create_training_data(self, integer_sequence: list[int]) -> tuple[Any, np.ndarray]:
        """Create the training data

        :param integer_sequence: training data as integer sequence
        :return: Training data
        """

        # Create the training data
        num_sequences = len(integer_sequence) - self.sequence_length

        inputs = []
        targets = []

        for i in range(num_sequences):
            inputs.append(integer_sequence[i : i + self.sequence_length])
            targets.append(integer_sequence[i + self.sequence_length])

        # One-Hot encode the inputs
        num_unique_symbols = len(self.mapping)
        one_hot_inputs = keras.utils.to_categorical(inputs, num_classes=num_unique_symbols)
        np_targets = np.array(targets)

        return one_hot_inputs, np_targets

    def __str__(self) -> str:
        text = []
        self.model.summary(print_fn=lambda x: text.append(x))
        return "\n".join(text)