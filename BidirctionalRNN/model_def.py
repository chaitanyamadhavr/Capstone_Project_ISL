import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense


num_classes = 61

model = tf.keras.Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(60,225)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(num_classes, activation='softmax'))
