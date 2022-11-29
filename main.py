# Imports
import RGB2GrayTransformer
import HogTransformer
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL

import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector

from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, Normalizer


# Rotate the images 5,10,15, and 20 degrees left and right to improve dataset
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
        i = 0
        for subdir in os.listdir(src):
            if subdir in include:
                print(f'Loaded "{subdir}"')
                current_path = os.path.join(src, subdir)

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


def plot_dataset(data):
    # Get all unique values in the list of labels
    labels = np.unique(data['label'])

    # setup matplotlib figure and axis
    fig, axes = plt.subplots(1, len(labels), figsize=(15, 4))

    # make a plot to show examples of each label from dataset
    for ax, label in zip(axes, labels):
        index = data['label'].index(label)
        ax.imshow(data['data'][index])
        ax.axis('off')
        ax.set_title(label)
    plt.show()  # Commented out until needed to store image for presentation


def plot_training_results(pass_score_dict, fail_score_dict):
    print('Reached plot_training_results()')

    # Plot performance
    plt.rcParams['figure.figsize'] = [7.5, 3.5]
    plt.rcParams['figure.autolayout'] = True

    # Pass Performance
    pass_score_dict = np.array(pass_score_dict)
    x = np.arange(0, len(pass_score_dict))
    y = pass_score_dict
    plt.plot(x, y, color="blue", label="Pass")

    # Fail performance
    fail_score_dict = np.array(fail_score_dict)
    x_fail = np.arange(0, len(fail_score_dict))
    y_fail = fail_score_dict
    plt.plot(x_fail, y_fail, color="red", label="Fail")

    # Customize Scatter Plot
    plt.title("SGD Classifier Accuracy")
    plt.xlabel("Number of Samples")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.show()  # Show the scatter plot


def get_classifier(data):
    print('Reached get_classifier().\nRetrieving Classification Model...')
    # Split data into test set and training set
    # Use 80% for training and remaining data for test-set
    # Use train_test_split since we have multiple 'categories'
    X = np.array(data['data'])
    y = np.array(data['label'])

    # Perform train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,  # 20% Test Size
        shuffle=True,  # Randomize samples
        random_state=100,
    )

    # Create instances of RGB2Gray and Hog Transformers
    grayify = RGB2GrayTransformer.RGB2GrayTransformer()
    hogify = HogTransformer.HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys'
    )

    scalify = StandardScaler()

    # Transform training data and testing data into usable data
    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)
    # Used for debugging / displaying numerical values of dataset
    # print(X_train_prepared)

    # Train the model using the Stochastic Gradient Descent (SGD) Classifier Model
    sgd_clf = SGDClassifier(random_state=100, max_iter=1000,
                            tol=1e-3)  # Random_state is used to recreate the 'random' training
    sgd_clf.fit(X_train_prepared, y_train)  # Train the model with the PREPARED dataset
    print(sgd_clf)
    # Setup Testing Data
    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    # Train the SGD ASL Recognition Model and Measure Performance
    y_pred = sgd_clf.predict(X_test_prepared)
    pass_count, fail_count = 0, 0
    pass_score_dict = []
    fail_score_dict = []
    for i in range(0, len(X_test_prepared)):
        if y_pred[i] == y_test[i]:
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1
        pass_score = pass_count / len(X_test_prepared)
        pass_score_dict.append(pass_score)

        fail_score = fail_count / len(X_test_prepared)
        fail_score_dict.append(fail_score)
    # Plot testing performance
    plot_training_results(pass_score_dict, fail_score_dict)

    print(np.array(y_pred == y_test)[:25])
    print('\nPercentage Correct: ', 100 * np.sum(y_pred == y_test) / len(y_test))
    joblib.dump(sgd_clf, 'model.sav')
    return sgd_clf


# Main Function
def main():
    print("Reached main()")

    # Load / Build Model
    model = None
    try:
        model = joblib.load('model.sav')
        return model
    except:
        print('Preparing Data...')
        # Set the filepath
        data_path = 'res/asl'
        os.listdir(data_path)

        # Include all the letter folders from ASL dataset
        include = {'nothing', 'A', 'B', 'C', 'D', 'E', 'F',
                   'del', 'G', 'H', 'I', 'J', 'K', 'L',
                   'space', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                   'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

        # Resize all images to 80px by 80px
        base_name = 'asl'
        width = 80

        # Resize / Prep All Images from Dataset (Only needs to be called if pkl hasn't been created)
        # rotate_all(src=data_path, include=include)
        data = resize_all(src=data_path, pklname=base_name, width=width, include=include)

        # Load data from disk and print summary of the data
        # data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
        print('Data prepared.')
        print('number of samples: ', len(data['data']))
        print('keys: ', list(data.keys()))
        print('image shape: ', data['data'][0].shape)
        print('labels:', np.unique(data['label']))
        model = get_classifier(data)
    print('Retrieved Classification Model.')

    print('Loading CV...')
    # Computer Vision
    stream = cv2.VideoCapture(0)
    stream.set(3, 640)
    stream.set(4, 480)

    grayify = RGB2GrayTransformer.RGB2GrayTransformer()
    hogify = HogTransformer.HogTransformer(
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys'
    )
    detector = HandDetector(detectionCon=0.5, maxHands=1)

    while True:
        print('CV Running...')
        data = dict()
        data['data'] = []

        # Capture frame from video
        success, img = stream.read()

        # Find the hand and it's landmarks
        im = detector.findHands(img)
        lmList, bbox = detector.findPosition(im)

        # Resize image and add to df
        im = resize(im, (80, 80, 3))
        data['data'].append(im)

        X_test_gray = grayify.transform(data['data'])
        X_test_hog = hogify.transform(X_test_gray)

        # print(f'X_test_prepared: {X_test_hog}')

        # Make a Prediction
        pred = model.predict(X_test_hog)
        print(pred)

        # Show Frame
        cv2.imshow('CV', img)
        cv2.waitKey(10)  # Update every 10 frames
    # Close program
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
