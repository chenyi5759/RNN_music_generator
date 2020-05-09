'''
Dataset generator

Authour: Chen-Yi Liu
Date: July 3, 2018
'''

import numpy as np
from random import randrange


class Dataset():
  def __init__(self, data, input_steps, output_steps, batch_size):
    self.data = np.frombuffer(data, dtype=np.int8)
    self.size = len(data) - input_steps - output_steps
    self.input_steps = input_steps
    self.num_features = 256
    self.output_steps = output_steps
    self.batch_size = batch_size
    
  def __len__(self):
    return self.size

  def __iter__(self):
    return self

  def __next__(self):
    x = np.zeros((self.batch_size, self.input_steps, self.num_features), \
                 dtype=np.float32)
    y = np.zeros((self.batch_size, self.num_features), \
                 dtype=np.float32)
    
    for i in range(self.batch_size): 
      start = randrange(self.size)
      input_end = start + self.input_steps
      #output_end = input_end + self.output_steps
      x[i, range(self.input_steps), self.data[start:input_end]] = 1
      y[i, self.data[input_end]] = 1
      
    return x, y

