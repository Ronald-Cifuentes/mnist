from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)

# Load the MNIST dataset
dataset, metadata = tfds.load("mnist", as_supervised=True, with_info=True)

# Obtain the 60,000 training examples
# Obtain the 10,000 test examples
train_dataset, test_dataset = dataset["train"], dataset["test"]

# Define simple labels for each possible output of our network
class_names = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
]

# Obtain the number of examples in variables
num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples


# Define a normalization function
# Normalize: numbers from 0 to 255, in a range from 0 to 1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# Call the normalization function on each data in both datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Define the structure of our network
# Specify that the input is 784 neurons, a square, that are 28x28 pixels
# Specify that two dense layers with 64 neurons and ReLU activation will be used
# Specify that the output are 10 neurons with softmax activation
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(14, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Functions to execute
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Batch Learning of 32 each batch
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Do the learning
model.fit(
    train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE)
)

# Evaluate our already trained model against the test dataset
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / 32)
)

print("Resultados de la prueba: ", test_accuracy)

for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("Prediccion: {}".format(class_names[predicted_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#888888")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


numrows = 5
numcols = 3
numimages = numrows * numcols

plt.figure(figsize=(2 * 2 * numcols, 2 * numrows))

for i in range(numimages):
    plt.subplot(numrows, 2 * numcols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(numrows, 2 * numcols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

plt.show()
