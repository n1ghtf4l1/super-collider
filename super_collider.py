__author__ = 'Anubhav De'

"""
Given a list of collision events and their properties, predict whether a τ → 3μ decay happened in the collision.
"""

# importing necessary libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d

from hep_ml.losses import BinFlatnessLossFunction, KnnAdaLossFunction, KnnFlatnessLossFunction
from hep_ml.nnet import MLPClassifier
from hep_ml import metrics

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# switches
DO_IMP, DO_BIN, DO_ADA, DO_KNNF = False, False, False, False

# loading data files
print("loading data files from ../super-collider/training.csv using pandas...")

train = pd.read_csv("../super-collider/training.csv")
train = train[train['min_ANNmuon'] > 0.4]

test = pd.read_csv("../super-collider/test.csv")

check_agreement = pd.read_csv("../super-collider/check_agreement.csv", index_col='id')
corr_check = pd.read_csv("../super-collider/check_correlation.csv")
signal = train.signal
trainids = train.index.values

# work in progress
