import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model

# Define the model
num_classes = 17  # Number of classes
input_shape = (50, 225)  # Adjust input shape as needed
video_input = Input(shape=input_shape)

# Convolutional layers to capture spatial features (1D Convolution)
conv1d_1 = Conv1D(32, kernel_size=3, activation='relu')(video_input)
maxpool1d_1 = MaxPooling1D(pool_size=2)(conv1d_1)
conv1d_2 = Conv1D(64, kernel_size=3, activation='relu')(maxpool1d_1)
maxpool1d_2 = MaxPooling1D(pool_size=2)(conv1d_2)

# LSTM layers to capture temporal features
lstm1 = Bidirectional(LSTM(64, return_sequences=True))(maxpool1d_2)
lstm2 = Bidirectional(LSTM(128, return_sequences=True))(lstm1)
lstm3 = Bidirectional(LSTM(64, return_sequences=False))(lstm2)
# TimeDistributed Dense layer to apply to each time step
#time_dist = TimeDistributed(Dense(64, activation='relu'))(lstm2)

# Classification layer
classification_output = Dense(num_classes, activation='softmax')(lstm3)

# Create the model
model = Model(inputs=video_input, outputs=classification_output)
