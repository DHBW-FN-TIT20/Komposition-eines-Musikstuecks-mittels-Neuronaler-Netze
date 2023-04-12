# This script creates a TensorFlow model with a neural network for the function y = 2x + 1
# The input data is a set of points (x,y) that ist generated with the function y = 2x + 1
# The model ist trained to find the y value for a given x value
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Helper libraries

math_function = lambda x: x * 2 + 1

# Generate input data
x_data_train = np.random.rand(1_000).astype(np.float32)  # Input data
y_data_train = math_function(x_data_train)  # Output data

# Generate test data
x_data_test = np.random.rand(100).astype(np.float32)
y_data_test = math_function(x_data_test)


# Build the model with 1 input layer and 1 output layer
model = keras.Sequential(
    [
        keras.layers.Dense(1, input_shape=(1,)),  # The input shape is 1 because the input data is a single value
        keras.layers.Dense(1),
    ],
)

# Compile the model
model.compile(optimizer="sgd", loss="mean_squared_error")

# Train the model
model.fit(x_data_train, y_data_train, epochs=10)

# Evaluate the model
loss = model.evaluate(x_data_test, y_data_test, verbose=1)
print("Test loss:", loss)

# Calcualte the accuracy
y_predicted = model.predict(x_data_test)
diffrence = 0
for i, prediction in enumerate(y_predicted):
    diffrence += abs(prediction - y_data_test[i])

print(f"Average diffrence: {diffrence / len(y_predicted)}")

# Predict the test data
while True:
    x_input = input("Enter a value between 0 and 1: ").strip()

    if x_input == "e":
        break

    x_input = float(x_input)

    y_predict = model.predict([x_input])
    y_calcualted = math_function(x_input)

    print(f"Predict:\nx = {x_input}, y = {y_predict}")
    print(f"Calculated:\nx = {x_input}, y = {y_calcualted}")
