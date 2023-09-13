import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import helper_funcs as helper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path


import warnings
warnings.filterwarnings('ignore')

# model analysis and development
import edgeimpulse as ei
from livelossplot import PlotLossesKeras 

# Import the necessary callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from livelossplot.outputs import MatplotlibPlot
from plot_keras_history import  plot_history
from sklearn.metrics import classification_report

# inferencing
from PIL import Image

# Set random seed for reproducibility
random_state = 42
tf.random.set_seed(random_state)
random.seed(random_state)
from datetime import datetime