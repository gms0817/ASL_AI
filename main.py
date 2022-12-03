# Imports
import os

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from google.protobuf.json_format import MessageToDict
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization


def rotate_all(src, include):
    # Read all images in the path and write to a new destination path
    for subdir in os.listdir(src):
        if subdir in include:
            print(f'Loaded "{subdir}"')
            current_path = os.path.join(src, subdir)

            # Rotate the images in the directory and make new files
            for file in os.listdir(current_path):
                image_path = current_path[0:7] + '/' + subdir + '/' + file
                new_image_path = current_path[0:7] + '/' + subdir + '/' + file[0:file.index('.')]
                if file[-3:] in {'jpg', 'png'}:
                    current_image = Image.open(image_path)
                    # Rotate images 5, 10, 15, and 20 degrees left and right
                    rotated_image = current_image.rotate(-5)  # Rotate -5 degrees
                    rotated_image.save(new_image_path + '-left-5.png')

                    rotated_image = current_image.rotate(-10)  # Rotate -10 degrees
                    rotated_image.save(new_image_path + '-left-5.png')

                    rotated_image = current_image.rotate(-15)  # Rotate -15 degrees
                    rotated_image.save(new_image_path + '-left-15.png')

                    rotated_image = current_image.rotate(-20)  # Rotate -20 degrees
                    rotated_image.save(new_image_path + '-left-20.png')

                    rotated_image = current_image.rotate(5)  # Rotate 5 degrees
                    rotated_image.save(new_image_path + '-right-5.png')

                    rotated_image = current_image.rotate(10)  # Rotate 10 degrees
                    rotated_image.save(new_image_path + '-right-5.png')

                    rotated_image = current_image.rotate(15)  # Rotate 15 degrees
                    rotated_image.save(new_image_path + '-right-15.png')

                    rotated_image = current_image.rotate(20)  # Rotate 20 degrees
                    rotated_image.save(new_image_path + '-right-20.png')
                    print(f'Current Path: {current_path} | File: {file}')
                    print(f'New Image Path: {new_image_path} | File: {file}')


# Read, resize, and store the data into a dictionary
def resize_all(src, pklname, include, width=150, height=None):
    print('Reached resize_all().')  # Configure pkl filename
    pklname = 'asl_80x80px.pkl'

    try:
        data = joblib.load('asl_80x80px.pkl')
        return data
    except:
        height = width

        # Configure Dictionary
        data = dict()
        data['label'] = []
        data['data'] = []

        # Read all images in the path and write to a new destination path
        for subdir in os.listdir(src):
            if subdir in include:
                print(f'Loaded "{subdir}"')
                current_path = os.path.join(src, subdir)
                i = 0
                # Populate the dictionary
                for file in os.listdir(current_path):
                    while i < 1000:
                        if file[-3:] in {'jpg', 'png'}:
                            im = imread(os.path.join(current_path, file))
                            im = resize(im, (width, height))
                            data['label'].append(subdir)
                            data['data'].append(im)
                        i = i + 1

            # Store the data
            joblib.dump(data, pklname)
        return data


def get_data(label):
    # Load the data
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'nothing', 'space', 'del']

    img_size = 224
    data = []

    for label in labels:
        path = os.path.join('res/asl', label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # Convert BGR to RGB
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshape images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)

    return np.array(data)


def get_cnn():
    # Attempt to load the model
    model = None
    try:
        model = joblib.load('model.sav')
        return model
    except:
        # Create model if model is not found
        img_size = 224

        # Prepare the data
        training_data = get_data('res/asl')
        validation_data = get_data('res/asl/asl-alphabet-test')
        print('Data prepared.')

        # Data Preprocessing and Data Augmentation
        x_train = []
        y_train = []
        x_validation = []
        y_validation = []

        for feature, label in training_data:
            x_train.append(feature)
            y_train.append(label)
        for feature, label in validation_data:
            x_validation.append(feature)
            y_validation.append(label)

        # Normalize the Data
        x_train = np.array(x_train) / 255
        x_validation = np.array(x_validation) / 255

        x_train.reshape(-1, img_size, img_size, 1)
        y_train = np.array(y_train)

        x_validation.reshape(-1, img_size, img_size, 1)
        y_validation = np.array(y_validation)

        # Data Augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.2,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        # Define the model
        classes = 29
        batch = 32
        epochs = 15
        learning_rate = 0.001

        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(classes, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(x_train, y_train,
                            batch_size=batch, epochs=epochs,
                            validation_data=(x_validation, y_validation),
                            shuffle=True, verbose=1)
        """
        # CNN - 3 Convolutional Layers followed by max pooling layers
        model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))  # Used to help avoid overfitting

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(29, activation="softmax"))  # Use 29 because we have 29 labels

        model.summary()

        # Compile the model using Adam optimizer
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train model for 500 epochs because of small learning rate
        history = model.fit(x_train, y_train,
                            epochs=5, batch_size=32, 
                            validation_data=(x_validation, y_validation),
                            shuffle=True, verbose=1) """

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(5)

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        # Save the model
        joblib.dump(model, 'model.sav')

        return model


def num_to_letter(pred):
    apred = ""
    if pred == 0:
        apred = 'A'
    elif pred == 1:
        apred = 'B'
    elif pred == 2:
        apred = 'C'
    elif pred == 3:
        apred = 'D'
    elif pred == 4:
        apred = 'E'
    elif pred == 5:
        apred = 'F'
    elif pred == 6:
        apred = 'G'
    elif pred == 7:
        apred = 'H'
    elif pred == 8:
        apred = 'I'
    elif pred == 9:
        apred = 'J'
    elif pred == 10:
        apred = 'K'
    elif pred == 11:
        apred = 'L'
    elif pred == 12:
        apred = 'M'
    elif pred == 13:
        apred = 'N'
    elif pred == 14:
        apred = 'O'
    elif pred == 15:
        apred = 'P'
    elif pred == 16:
        apred = 'Q'
    elif pred == 17:
        apred = 'R'
    elif pred == 18:
        apred = 'S'
    elif pred == 19:
        apred = 'T'
    elif pred == 20:
        apred = 'U'
    elif pred == 21:
        apred = 'V'
    elif pred == 22:
        apred = 'W'
    elif pred == 22:
        apred = 'X'
    elif pred == 23:
        apred = 'Y'
    elif pred == 24:
        apred = 'Z'
    elif pred == 25:
        apred = 'nothing'
    elif pred == 26:
        apred = 'del'
    elif pred == 27:
        apred = 'space'

    return str(apred)


# Main Function
def main():
    print("Reached main()")

    # Load / Build Model
    model = None
    try:
        model = joblib.load('model.sav')
        return model
    except Exception as e:
        print('Preparing Data...')
        # Set the filepath
        data_path = 'res/asl'
        os.listdir(data_path)

        # Load the model
        model = get_cnn()
    print('Retrieved Classification Model.')

    print('Loading CV...')
    # Computer Vision
    stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    stream.set(3, 740)
    stream.set(4, 580)

    detector = HandDetector(detectionCon=0.5, maxHands=1)

    while True:
        print('CV Running...')
        data = []

        # Capture frame from video
        success, img = stream.read()

        # Find hands and put bbox on hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        # Check which hand is on screen
        processed_img = img  # Used to not show the user the potentially flipped image
        handType = detector.handType()
        if lmList and handType is not None:

            if handType == 'Left':
                # Put hand label on image
                cv2.putText(img, 'Left Hand', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

                # Flip the image
                processed_img = cv2.flip(img, 1)

            if handType == 'Right':
                # Put hand label on image
                cv2.putText(img, 'Right Hand', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

            # Convert BGR to RGB img array
            img_arr = processed_img[..., ::-1]

            # Resize image array and store image data into numpy array
            resized_arr = cv2.resize(img_arr, (224, 224))
            data.append(resized_arr)
            data = np.array(data) / 255
            data.reshape(-1, 224, 224, 1)

            # Make a Prediction
            pred = model.predict(data)
            pred = pred.reshape(1, -1)[0]
            pred_classes = np.argmax(pred, axis=0)

            # Convert num to letter
            apred = num_to_letter(pred_classes)
            print(apred)

            # Put prediction on image
            cv2.putText(img, str(apred), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Show Frame
        cv2.imshow('CV', img)
        cv2.waitKey(1)  # Update every 10 frames

    # Close program
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
