model = keras.models.Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
