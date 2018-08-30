import os
import csv
import cv2
import numpy as np
from keras.layers import *
from keras.models import Sequential

RECORD_IMG_FOLDER = 'IMG'
RECORDS_MAIN_DIR = 'Records'
RECORD_CSV_FILE = 'driving_log.csv'
MODEL_SAVE_FILE = 'model.h5'
STEERING_CORRECTION = 0.2


#class RecordEntry:
#    def __init__(self, img, steering, record_dir):
#        self.img = img
#        self.steering = steering
#        self.img_dir = record_dir


record_dirs = [os.path.join(RECORDS_MAIN_DIR, name) for name in os.listdir(RECORDS_MAIN_DIR) if os.path.isdir(os.path.join(RECORDS_MAIN_DIR, name))]


images = []
steerings = []

for record_dir in record_dirs:
    csv_file = os.path.join(record_dir, RECORD_CSV_FILE)
    with open(csv_file) as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header
        for row in reader:
            img = cv2.imread(os.path.join(record_dir, row[0]))
            images.append(img)
            images.append(cv2.flip(img, 1))
            #images.append(cv2.imread(os.path.join(record_dir, row[1])))
            #images.append(cv2.imread(os.path.join(record_dir, row[2])))

            steering = float(row[3])
            steerings.append(steering)
            steerings.append(-steering)
            #steerings.append(steering + STEERING_CORRECTION)
            #steerings.append(steering - STEERING_CORRECTION)


X_train = np.array(images)
y_train = np.array(steerings)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPool2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save(MODEL_SAVE_FILE)
