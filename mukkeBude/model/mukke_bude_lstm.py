import os
from typing import Any
from typing import List
from typing import Tuple

import keras
import numpy as np
from mukkeBude.mapping import MusicMapping


class MukkeBudeLSTM:
    def __init__(
        self,
        mapping: MusicMapping,
        hidden_layer: List[int] = [256],
        loss="sparse_categorical_crossentropy",
        activation="softmax",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        sequence_length: int = 64,
        model: keras.Model = None,
    ) -> None:
        """LSTM model for MukkeBude

        :param mapping: Dictionary mapping unique symbols to integers
        :param hidden_layer: Number of hidden Layers, defaults to [256]
        :param loss: Loss function of the LSTM, defaults to "sparse_categorical_crossentropy"
        :param activation: Activation function of the LSTM, defaults to "softmax"
        :param optimizer: optemizer of the LSTM, defaults to keras.optimizers.Adam(learning_rate= 0.001)
        :param sequence_length: Length of the sequences, defaults to 64
        :param model: Pretrained model, defaults to None. If a model is provided, the other parameters are ignored
        """
        self.mapping = mapping

        if model is not None:
            self.model = model
            return

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

    def train(
        self,
        dataset: List[int],
        epochs: int = 50,
        batch_size: int = 64,
        tensorboard_callback: keras.callbacks.TensorBoard = None,
    ) -> None:
        """Train the LSTM model

        :param dataset: Training dataset
        :param epochs: Number of epochs to train, defaults to 10
        :param batch_size: Size of the batches, defaults to 64
        :param tensorboard_callback: There you can pass a TensorBoard callback, defaults to None
        """
        inputs, targets = self.__create_training_data(dataset)

        # Only allow TenosrBoard callback
        if tensorboard_callback is not None and not isinstance(tensorboard_callback, keras.callbacks.TensorBoard):
            raise TypeError("Only TensorBoard callback allowed")

        self.model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback])

    def generate(
        self,
        start_seed: str,
        max_length: int = 128,
        max_sequence_length: int = 64,
        temperature=0.8,
    ) -> List[str]:
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
        output_melody.extend(start_seed.split(" "))

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
            if output_symbol == "/" or output_symbol == "xxeos":
                break

            # Update the output melody
            output_melody.append(output_symbol)

        return output_melody

    def save(self, name: str) -> str:
        """Save the model with the given name. The model will be saved in the `model/preTrainedModels` folder.

        :param name: Name of the model
        :return: Path to the saved model
        """
        path = os.path.join(os.path.dirname(__file__), "preTrainedModels", name)
        self.model.save(path)
        return path

    @staticmethod
    def load(mapping: MusicMapping, name: str) -> "MukkeBudeLSTM":
        """Load the model with the given name from the `model/preTrainedModels` folder.

        :param mapping: Dictionary mapping unique symbols to integers
        :param name: Name of the model
        :return: Loaded model
        """
        path = os.path.join(os.path.dirname(__file__), "preTrainedModels", name)
        model = keras.models.load_model(path)
        return MukkeBudeLSTM(mapping=mapping, model=model)

    def __create_training_data(self, integer_sequence: List[int]) -> Tuple[Any, np.ndarray]:
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
