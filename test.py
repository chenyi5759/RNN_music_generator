'''
RNN Music Generator

Authors: Chen-Yi
Date: 2018.07.02
'''

import wave
from keras.models import load_model
import numpy as np

from import_data import Dataset


TIME_STEPS = 100
OUTPUT_CATEGORIES = 256
MODEL_FILE = "model.h5"
DATASET_FILE = "ghibli"
RESULT_FILE = "result.wav"


# Load the trained model
model = load_model(MODEL_FILE)


# Load and process dataset
print("Preparing dataset...")
with open(DATASET_FILE, 'rb') as dataset_file:
  data = dataset_file.read()
training_set = Dataset(data, TIME_STEPS, 1, 1)


# Test the network
print("Testing the network...")
starter, _ = next(iter(training_set))
result = list(np.argmax(starter[0], axis=1))
for i in range(10000):
  prediction = model.predict_on_batch(starter)
  next_sample = np.random.choice(OUTPUT_CATEGORIES, p=prediction[0])
  result.append(next_sample)
  starter[0, :TIME_STEPS-1, :] = starter[0, -TIME_STEPS+1:, :]
  starter[0, -1, :] = 0
  starter[0, -1, next_sample] = 1

  if i%1000 == 0:
    print(f"Rendering {i}th sample...")


print("Saving result...")
wave_file = wave.open(RESULT_FILE, mode='wb')
wave_file.setnchannels(1)
wave_file.setsampwidth(1)
wave_file.setframerate(11025)
wave_file.writeframes(bytearray(result))
wave_file.close()

