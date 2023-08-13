import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="tanh", input_shape=(784,)),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    loss=tf.losses.MSE,
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"])
model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=50,
    validation_data=(x_test, y_test),
    verbose=2,
)

res = model.evaluate(x_test, y_test, verbose=0)
print("정확률=", res[1] * 100)
