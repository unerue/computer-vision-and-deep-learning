import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
# dmlp.add(Dense(units=1024, activation="relu", input_shape=(784,)))
# dmlp.add(Dense(units=512, activation="relu"))
# dmlp.add(Dense(units=512, activation="relu"))
# dmlp.add(Dense(units=10, activation="softmax"))

model.compile(
    loss=tf.losses.categorical_crossentropy,
    # loss="categorical_crossentropy",
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    # metrics=["accuracy"],
    metrics=[tf.metrics.categorical_accuracy],
)
hist = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=50,
    validation_data=(x_test, y_test),
    verbose=2,
)
print("정확률=", model.evaluate(x_test, y_test, verbose=0)[1] * 100)

model.save("dmlp_trained.h5")

import matplotlib.pyplot as plt

plt.plot(hist.history["categorical_accuracy"])
plt.plot(hist.history["val_categorical_accuracy"])
plt.title("Accuracy graph")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["train", "test"])
plt.grid()
plt.show()

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Loss graph")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train", "test"])
plt.grid()
plt.show()
