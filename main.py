# Imports
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.transform import rescale
from collections import Counter


# Read, resize, and store the data into a dictionary
def resize_all(src, pklname, include, width=150, height=None):
    height = height if height is not None else width

    # Configure Dictionary
    data = dict()
    data['description'] = 'resized ({0}x{1})asl hand-signs in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    # Configure pkl filename
    pklname = f'{pklname}_{width}x{height}px.pkl'
    # Read all images in the path and write to a new destination path
    for subdir in os.listdir(src):
        if subdir in include:
            print(f'Loaded "{subdir}"')
            current_path = os.path.join(src, subdir)

            # Populate the dictionary
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)

        # Store the data
        joblib.dump(data, pklname)


# Main Function
def main():
    print("Reached main()")
    # Set the filepath
    data_path = 'res/asl'
    os.listdir(data_path)

    # Include all the letter folders from ASL dataset
    include = {'A', 'B', 'C', 'D', 'E', 'F',
               'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S',
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

    # Resize all images to 80px by 80px
    base_name = 'asl'
    width = 80

    # Resize / Prep All Images from Dataset (Only needs to be called if pkl hasn't been created)
    resize_all(src=data_path, pklname=base_name, width=width, include=include)

    # Load data from disk and print summary of the data
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')

    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))

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
    plt.show()

    # Split data into test set and training set
    # Use 80% for training and remaining data for test-set
    # Use train_test_split since we have multiple 'categories'
    X = np.array(data['data'])
    y = np.array(data['label'])

    # Perform train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 20% Test Size
        shuffle=True,  # Randomize samples
        random_state=42,
    )

    # Transform image into HOG


if __name__ == "__main__":
    main()
