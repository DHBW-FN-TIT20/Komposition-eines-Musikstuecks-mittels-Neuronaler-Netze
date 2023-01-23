# This script creates a TesnorFlow model for linear regression
# The input data is a set of points (x,y) that ist generated with the function y = 2x + 1
# The model ist trained to find the y value for a given x value

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate input data
x_data_train = np.random.randint(0, 10_000, 1_000).astype(np.float32)
y_data_train = x_data_train * 2 + 1

# Generate test data
x_data_test = np.random.randint(0, 10_000, 1_000).astype(np.float32)
y_data_test = x_data_test * 2 + 1

# Show the train data
# plt.plot(x_data_train, y_data_train, 'ro', label='Original data')
# plt.legend()
# plt.show()

# Convert input data to dictionary
x_data_train = {'x': x_data_train}
x_data_test = {'x': x_data_test}

# Create the input function for the training
def make_input_fn(x, y, num_epochs=10, shuffle=True, batch_size=32):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x, y))    # create a dataset from the input data that is needed for the model
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset
    return input_fn

train_input_fn = make_input_fn(x_data_train, y_data_train)
test_input_fn = make_input_fn(x_data_test, y_data_test, num_epochs=1, shuffle=False)

# Create feature columns
feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]

# Create the estimator
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Train the model
estimator.train(train_input_fn)

# Predict the test data
y_predicted = estimator.predict(test_input_fn)

# Round the predictions
# y_predicted_rounded = []
# for prediction in y_predicted:
#     y_predicted_rounded.append(round(prediction['predictions'][0]))

# Calculate the accuracy
correct = 0
wrong = 0
diffrence = 0
for i, prediction in enumerate(y_predicted):
    if prediction["predictions"][0] == y_data_test[i]:
        correct += 1
        # print('Prediction: ', prediction, ' - Correct')
    else:
        wrong += 1
        # print('Prediction: ', prediction, ' - Wrong')
    
    diffrence += abs(prediction["predictions"][0] - y_data_test[i])

print('Correct: ', correct)
print('Wrong: ', wrong)
print('Accuracy: ', correct / (correct + wrong))
print('Average diffrence: ', diffrence / (correct + wrong))

# # Show the test data
# plt.plot(x_data_test['x'], y_data_test, 'ro', label='Original data')

# # Show the predicted data
# for i, prediction in enumerate(y_predicted):
#     plt.plot(x_data_test['x'][i], prediction['predictions'][0], 'bo', label='Predicted data')

# plt.show()

# Evaluate the model
eval_result = estimator.evaluate(test_input_fn)
print("Evaluation result: ", eval_result)