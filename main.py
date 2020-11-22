import warnings
import numpy as np
from Functions import DataPreprocessing, ModelPrediction

warnings.filterwarnings("ignore")
np.random.seed(40)

# Preparation Data for Prediction
DataPreprocessing()
# Model Prediction and Data Visualization
ModelPrediction()
