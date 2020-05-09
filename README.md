# RNN Music Generator
This deep learning model learns to predict the next frame following a series of audio frames taken from an audio file.
This is an example of a time series prediction model. The model uses a combination of one dimensional dilated convolution and Long Shor Term Memory (LSTM). The model is inspired by the [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper by DeepMind.

### Usage:
#### Training the model
The script [train.py](train.py) takes a single channel mu-law encoded raw audio file as training dataset, and saves the trained model.

#### Running the model
The script [test.py](test.py) uses the trained model and a brief fragment of an audio clip and generates an audio clip based on the sound in the brief fragment.


