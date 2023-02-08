# This script creates a TensorFlow model with a neural network for the function y = 2x + z * k - 20
# The input data is a set of points (x,y) that ist generated with the function y = 2x + z * k - 20
# The model ist trained to find the y value for a given x value

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

math_function = lambda x, z, k: x * 2 + z * k - 20

# Generate input data
train_data = np.random.rand(10_000, 3).astype(np.float32)  # Input data
y_train_data = math_function(train_data[:, 0], train_data[:, 1], train_data[:, 2])  # Output data

# Generate test data
test_data = np.random.rand(100, 3).astype(np.float32)
y_test_data = math_function(test_data[:, 0], test_data[:, 1], test_data[:, 2])

# Build the model with 1 input layer and 1 output layer
model = keras.Sequential(
    [
        keras.layers.Dense(3, input_shape=(3,)),  # The input shape is 3 because the input data is x, z and k
        # keras.layers.Dense(20, activation="relu"),  # Hidden layer with 20 neurons
        keras.layers.Dense(1, activation="softmax"),  # Output layer with 1 neuron
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(train_data, y_train_data, epochs=10)

# Evaluate the model
loss = model.evaluate(test_data, y_test_data, verbose=1)
print("Test loss:", loss)

# Calcualte the accuracy
y_predicted = model.predict(test_data)
diffrence = 0
for i, prediction in enumerate(y_predicted):
    diffrence += abs(prediction - y_test_data[i])

print(f"Average diffrence: {diffrence / len(y_predicted)}")
