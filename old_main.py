# Imports
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
import time
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

warnings.simplefilter(action='ignore', category=FutureWarning)


# Build CNN Model
def get_cnn():
    # Attempt to load the model
    model = None
    try:
        model = load_model('model.h5')
        return model
    except:
        # Create model if model is not found
        # Load test and training data
        train_path = 'res/train'
        test_path = 'res/test'

        train_batches = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
            directory=train_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)
        test_batches = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
            directory=test_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)

        # Design the CNN
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(26, activation="softmax"))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        history = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches)

        # For getting next batch of testing imgs...
        imgs, labels = next(test_batches)

        scores = model.evaluate(imgs, labels, verbose=0)
        print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

        model.save('model.h5')

        # Plot the model performance over epochs
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(10)

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

        return model


# Calculate background average weights
def cal_accum_avg(frame, accumulated_weight):
    global background

    # Check if there is a background
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


# Find hand and contour it
def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours for the image
    # image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        return thresholded, hand_segment_max_cont


# Main Function
def main():
    print("Reached main()")

    # Load / Build Model
    print('Getting Classification Model...')
    model = get_cnn()

    print('Retrieved Classification Model.')
    print('Loading CV...')
    # Computer Vision
    stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Setup variables
    global background
    background = None
    accumulated_weight = 0.5
    ROI_top = 100
    ROI_bottom = 300
    ROI_right = 150
    ROI_left = 350
    num_frames = 0

    # Setup Word Dict
    word_dict = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                 'del', 'space']

    while True:
        print('CV Running...')
        ret, frame = stream.read()
        # flipping the frame to prevent inverted image of captured
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        # Get the background
        if num_frames < 70:

            cal_accum_avg(gray_frame, accumulated_weight)

            cv2.putText(frame_copy, "Scanning Background... Please wait.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        else:
            # segmenting the hand region
            hand = segment_hand(gray_frame)

            # Checking if we are able to detect the hand...
            if hand is not None:
                thresholded, hand_segment = hand
                # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
                                                              ROI_top)], -1, (255, 0, 0), 1)

                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded,
                                           cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded,
                                         (1, thresholded.shape[0], thresholded.shape[1], 3))

                # Make prediction
                pred = model.predict(thresholded)
                cv2.putText(frame_copy, f'Prediction: {word_dict[np.argmax(pred)]}',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(np.argmax(pred))

        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
                                                        ROI_bottom), (255, 128, 0), 3)
        # incrementing the number of frames for tracking
        num_frames += 1

        # Display the frame with segmented hand
        cv2.imshow("ASL Recognition", frame_copy)

        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Release the camera and destroy all the windows
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
