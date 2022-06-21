#https://github.com/jayaneetha/GenderClassifierLibriSpeech

#dataset link:   http://www.openslr.org/12/

#dataset name: 'train-clean-100'

import os
#host = socket.gethostname()
host ="asimov"
CLASSES = []
NUM_MFCC = 40
NUM_FRAMES = 87
DURATION = 2  # in seconds
GENDER_CLASSES = ['M', 'F']
PICKLE_FILE_PREFIX = 'LibriSpeech-mfcc-'

# environment specific constants
DATASET_STR = 'train-clean-100'
DATA_ROOT = r'D:\data\train-clean-100\LibriSpeech/'
DATA_DIR = os.path.join(DATA_ROOT, DATASET_STR)+ '/'
SPEAKER_FILE = DATA_ROOT + 'SPEAKERS.TXT'
SPEAKER_IDX = 0
CHAPTER_IDX = 1
FILENAME_IDX = 2
NUM_CLASSES = 251
    
    
import glob
import pickle

import librosa
import numpy as np
import os
import random
from tensorflow.keras.utils import to_categorical


speaker_gender_map = {}
TEMP_CLASS_INDEX = []


def init_speaker_gender_map():
    with open(SPEAKER_FILE) as f:
        content = f.readlines()

    for line in content:
        print(line)
        if DATASET_STR in line:
            sp = line.split('|')
            sp_id = sp[0].strip()
            gender = sp[1].strip()
            speaker_gender_map[sp_id] = gender


init_speaker_gender_map()


def get_speaker_ids():
    speaker_ids = []
    with open(SPEAKER_FILE) as f:
        content = f.readlines()

    for line in content:
        if DATASET_STR in line:
            sp = line.split('|')
            sp_id = sp[0].strip()
            speaker_ids.append(sp_id)

    return speaker_ids



def get_dataset():
    """@:param class_type: str - type of class needed to be in Y.
        values { 'gender' , 'speaker' }
    """
    train = []
    test = []
    valid = []
    speaker_ids = get_speaker_ids()
    num_classes_=[]
    for s in speaker_ids:
        file_list = glob.glob(DATA_DIR + s + '/*/*.flac')
        print("Loading Data from :", file_list, DATA_DIR + s)
        all_data = []
        for f in file_list:
            print("1111111111", f)
            f=f.split("/")[-1]
            speaker_id = f.split("\\")[SPEAKER_IDX]
            chapter_id = f.split("\\")[CHAPTER_IDX]
            filename = f.split("\\")[FILENAME_IDX]
    
            all_data.append(
                os.path.join(DATA_DIR, os.path.join(speaker_id, os.path.join(chapter_id, os.path.join(filename)))))
    
        random.shuffle(all_data)
        split_tuple = np.split(np.array(all_data), [int(0.7 * len(all_data)), int(0.9 * len(all_data))])
        train = train + split_tuple[0].tolist()
        test = test + split_tuple[1].tolist()
        valid = valid + split_tuple[2].tolist()
    
        x_train, y_train = get_XY_gender(train)
        x_test, y_test = get_XY_gender(test)
        x_valid, y_valid = get_XY_gender(valid)
        num_classes = len(GENDER_CLASSES)
        num_classes_.append(num_classes)

  

    return (x_train, to_categorical(y_train, num_classes=num_classes)), \
           (x_test, to_categorical(y_test, num_classes=num_classes)), \
           (x_valid, to_categorical(y_valid, num_classes=num_classes))


def get_XY_gender(fileList):
    x = []
    y = []

    random.shuffle(fileList)

    for f in fileList:
        #print(f)
        f_=f.split("/")[-1]
        speaker_id = f_.split("\\")[SPEAKER_IDX]
        x.append(f)
        g = GENDER_CLASSES.index(speaker_gender_map.get(speaker_id))
        y.append(g)

    return x, y



def load_from_pkl(filename):
    filename = PICKLE_FILE_PREFIX + filename
    print("loading from pickle file : {}".format(filename))
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

def load_wav(filename):
    audio, sr = librosa.load(filename, duration=DURATION)
    return audio, sr


def add_missing_padding(audio, sr):
    signal_length = DURATION * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio

def get_mfcc(filename):
    audio, sr = load_wav(filename)
    signal = add_missing_padding(audio, sr)
    return librosa.feature.mfcc(signal, sr, n_mfcc=NUM_MFCC)



def get_mfccs(file_list=False, pickle_file=False):
    if pickle_file:
        x_audio = load_from_pkl(pickle_file)
        return x_audio
    else:
        x_audio = []
        for i in range(len(file_list)):
            if i % 100 == 0:
                print("{0:.2f} loaded ".format(i / len(file_list)))
            x_audio.append(np.reshape(get_mfcc(file_list[i]), [NUM_MFCC, NUM_FRAMES, 1]))
        return x_audio

def save_to_pkl(data, filename):
    filename = PICKLE_FILE_PREFIX + filename
    print("Storing {} data to file: {}".format(len(data), filename))
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense, TimeDistributed, Conv1D, \
    MaxPooling1D, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential

from util import write_history

def model_v2(top_layer_units):
    model = Sequential()

    model.add(TimeDistributed(
        Conv1D(filters=16, kernel_size=4, padding='same', activation=tf.nn.relu, data_format='channels_last'),
        input_shape=(NUM_MFCC, NUM_FRAMES, 1)))

    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, padding='same', activation=tf.nn.relu)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=512, activation=tf.nn.tanh))
    model.add(Dense(units=256, activation=tf.nn.tanh))
    model.add(Dense(units=top_layer_units, activation=tf.nn.softmax, name='top_layer'))
    return model


#trainh test split
(X_train, y_train), (X_test, y_test), (X_valid, y_valid) = get_dataset()

#extract featrures from audio .flac file
x_audio_training = get_mfccs(X_train)
x_audio_validation = get_mfccs(X_valid)
x_audio_testing = get_mfccs(X_test)

#store features

feature_store = True  # save the feature pkl file


if feature_store:
    save_to_pkl(x_audio_training, 'training-gender-x.pkl')
    save_to_pkl(x_audio_validation, 'validation-gender-x.pkl')
    save_to_pkl(x_audio_testing, 'testing-gender-x.pkl')
    save_to_pkl(y_train, 'training-gender-y.pkl')
    save_to_pkl(y_valid, 'validation-gender-y.pkl')
    save_to_pkl(y_test, 'testing-gender-y.pkl')
    
    
    
    
    
    
#train model
    
print("Training length: {}".format(len(x_audio_training)))
print("Validation length: {}".format(len(x_audio_validation)))
print("Testing length: {}".format(len(x_audio_testing)))

model = model_v2(y_train.shape[1])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])  # optimizer was 'Adam'

model.summary()

x_train = np.reshape(x_audio_training, [len(x_audio_training), NUM_MFCC, NUM_FRAMES, 1])
x_valid = np.reshape(x_audio_validation, [len(x_audio_validation), NUM_MFCC, NUM_FRAMES, 1])

print("Start Fitting")
history = model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1, validation_data=(x_valid, y_valid))

model_name = 'Libri_Gender_v3.2'
print("Saving model as {}".format(model_name))
model.save_weights(model_name + '.h5')

write_history(history, filename='history-' + model_name + '.csv')



#test model
import pandas as pd
from collections import defaultdict
predict_df =defaultdict(list)
def test(x_audio_testing, y_test, model):
    correct_count = 0
    print("Testing on {} datasets".format(len(x_audio_testing)))
    for i in range(len(x_audio_testing)):
        audio = np.reshape(x_audio_testing[i], [1, NUM_MFCC, NUM_FRAMES, 1])
        predict_index = np.argmax(model.predict(audio))
        true_index = np.argmax(y_test[i])
        predict_df['predict_index'].append(predict_index)
        predict_df['true_index'].append(true_index)
        if predict_index == true_index:
            correct_count += 1

    test_accuracy = (correct_count / len(x_audio_testing) * 100)
    print("Test Accuracy: {}".format(test_accuracy))
    
    
test(x_audio_testing, y_test, model)
predict_df_=pd.DataFrame(predict_df)



prediction_class = predict_df_['predict_index'].map({0:"f", 1:"m"})
print(prediction_class)

