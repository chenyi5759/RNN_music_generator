'''
RNN Music Generator

Authors: Chen-Yi
Date: 2018.07.02
'''

from keras.models import Sequential, load_model
from keras.layers import Conv1D, LSTM, Dense, TimeDistributed

from import_data import Dataset

USE_PRETRAINED_MODEL = False
PRETRAINED_MODEL = "model.h5"

NUM_FILTERS = 64
KERNEL_SIZE = 2
TIME_STEPS = 1000
DILATION_RATE = 2
NUM_CONV_LAYERS = 3
OUTPUT_CATEGORIES = 256
BATCH_SIZE = 16
EPOCHS = 1
MODEL_FILE = "model.h5"
DATASET_FILE = "ghibli"


if USE_PRETRAINED_MODEL:
  model = load_model(PRETRAINED_MODEL)
else:
  print("Building the network...")
  model = Sequential()
  model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='causal', \
                   activation='relu', input_shape=(None, OUTPUT_CATEGORIES)))
  for i in range(1, NUM_CONV_LAYERS):
    model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='causal', \
                     dilation_rate=DILATION_RATE**i, activation='relu'))
  #model.add(LSTM(NUM_FILTERS, return_sequences=True))
  model.add(LSTM(NUM_FILTERS))
  #model.add(TimeDistributed(Dense(OUTPUT_CATEGORIES, activation='softmax')))
  model.add(Dense(OUTPUT_CATEGORIES, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')


print("Preparing dataset...")
with open(DATASET_FILE, 'rb') as dataset_file:
  data = dataset_file.read()
training_set = Dataset(data, TIME_STEPS, TIME_STEPS, BATCH_SIZE)


print("Training the network...")
model.fit_generator(training_set, epochs=EPOCHS, \
                    steps_per_epoch=len(training_set)/10000)


print("Saving model...")
model.save(MODEL_FILE)

