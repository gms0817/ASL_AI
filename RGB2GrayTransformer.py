# Imports
import skimage
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# Covert array of RGB Images to grayscale
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([skimage.color.rgb2gray(img) for img in X])
