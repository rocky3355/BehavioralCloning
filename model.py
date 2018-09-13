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
STEERING_CORRECTION = 0.1
DATA_CHUNK_SIZE = 1000


#class RecordEntry:
#    def __init__(self, img, steering, record_dir):
#        self.img = img
#        self.steering = steering
#        self.img_dir = record_dir


def load_data():
    idx = 0
    images = []
    steerings = []
    last = False

    record_dirs = [os.path.join(RECORDS_MAIN_DIR, name) for name in os.listdir(RECORDS_MAIN_DIR) if
                   os.path.isdir(os.path.join(RECORDS_MAIN_DIR, name))]
    for record_dir in record_dirs:
        print('Loading images from record folder: "{0}"'.format(record_dir))
        csv_file = os.path.join(record_dir, RECORD_CSV_FILE)
        with open(csv_file) as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header
            for row in reader:
                img = cv2.imread(os.path.join(record_dir, row[0]))
                images.append(img)
                images.append(cv2.flip(img, 1))
                images.append(cv2.imread(os.path.join(record_dir, row[1].strip())))
                images.append(cv2.imread(os.path.join(record_dir, row[2].strip())))

                steering = float(row[3])
                steerings.append(steering)
                steerings.append(-steering)
                steerings.append(steering + STEERING_CORRECTION)
                steerings.append(steering - STEERING_CORRECTION)

                idx += 1
                if idx % DATA_CHUNK_SIZE == 0:
                    yield images, steerings
                    images.clear()
                    steerings.clear()

    if not last:
        last = True
        yield images, steerings



def create_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(65, 320, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


model = create_model()
data_loader = load_data()

while 1:
    try:
        images, steerings = next(data_loader)
    except StopIteration:
        break
    X_train = np.array(images)
    y_train = np.array(steerings)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save(MODEL_SAVE_FILE)
