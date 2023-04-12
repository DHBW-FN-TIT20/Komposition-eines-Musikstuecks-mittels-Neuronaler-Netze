# This script create a neural network to classify the position of a point
# in a 2D plane. The point is either in the positive or negative half-plane.
# The network is trained with the backpropagation algorithm.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create a dataset of 1000 points in the 2D plane
# The points are either in the positive or negative half-plane
# The label is 1 if the point is in the positive half-plane
# The label is 0 if the point is in the negative half-plane
# The points are randomly generated
# The points are normalized to be in the range [-1, 1]
# The labels are one-hot encoded
x = np.random.uniform(-1, 1, (1000, 2))
y = np.zeros((1000, 2))
for i in range(1000):
    if x[i, 0] > 0:
        y[i, 1] = 1
    else:
        y[i, 0] = 1

# The dataset is split into training and test sets
# The training set contains 800 points
# The test set contains 200 points
x_train = x[0:800, :]
y_train = y[0:800, :]
x_test = x[800:1000, :]
y_test = y[800:1000, :]

# Show the training set
plt.figure()
plt.plot(x_train[:, 0], x_train[:, 1], 'b.')
plt.title('Training set')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Create a neural network with 2 inputs, 2 hidden layers with 10 neurons each,
# and 2 outputs
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])

# Compile the neural network
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the neural network
model.fit(x_train, y_train, epochs=10)

# Evaluate the neural network on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Show the test set
plt.figure()
plt.plot(x_test[:, 0], x_test[:, 1], 'b.')
plt.title('Test set')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Show the decision boundary
plt.figure()
plt.plot(x_test[:, 0], x_test[:, 1], 'b.')
plt.title('Decision boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
X = np.zeros((10000, 2))
X[:, 0] = X1.reshape(10000)
X[:, 1] = X2.reshape(10000)
Y = model.predict(X)
Y = Y[:, 1].reshape(100, 100)
plt.contour(X1, X2, Y, levels=[0.5])
plt.show()
