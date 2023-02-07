import keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH, SINGLE_FILE_DATASET

OUTPUT_UNITS = 44
NUM_UNITS = [256]
EPOCHS = 60
BATCH_SIZE = 64 # amount of samples before running backpropagation
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
SAVE_MODEL_PATH = "model_bach_part2.h5"


def build_model(output_units, num_units, loss, learning_rate):

    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)
    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=["accuracy"])
    model.summary()
    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    # generate the sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH, SINGLE_FILE_DATASET, MAPPING_PATH)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)



if __name__ == "__main__":
    train()